import csv
import itertools
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


def confusion_matrix_np(y_true, y_pred, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        true_label = int(true_label)
        pred_label = int(pred_label)
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[true_label, pred_label] += 1
    return cm


def compute_icbhi_metrics(y_true, y_pred, num_classes=4):
    """
    Compute ICBHI challenge metrics.

    Sensitivity: abnormal classes correctly predicted as abnormal.
    Specificity: normal class correctly predicted as normal.
    ICBHI score: average of sensitivity and specificity.
    """
    cm = confusion_matrix_np(y_true, y_pred, num_classes=num_classes)

    abnormal_total = cm[1:, :].sum()
    abnormal_correct = cm[1:, 1:].sum()
    sensitivity = abnormal_correct / abnormal_total if abnormal_total > 0 else 0.0

    normal_total = cm[0, :].sum()
    normal_correct = cm[0, 0]
    specificity = normal_correct / normal_total if normal_total > 0 else 0.0

    score = (sensitivity + specificity) / 2.0

    per_class_recall = {}
    for class_id in range(num_classes):
        total = cm[class_id, :].sum()
        per_class_recall[class_id] = cm[class_id, class_id] / total if total > 0 else 0.0

    false_negatives_as_normal = {}
    for class_id in range(1, num_classes):
        false_negatives_as_normal[class_id] = int(cm[class_id, 0])

    normal_as_abnormal = int(cm[0, 1:].sum())

    return {
        "confusion_matrix": cm,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "score": float(score),
        "per_class_recall": per_class_recall,
        "false_negatives_as_normal": false_negatives_as_normal,
        "normal_as_abnormal": normal_as_abnormal,
    }


def predict_with_thresholds(probabilities, thresholds):
    """
    Predict using per-class thresholds over softmax probabilities.

    If no class reaches its threshold, fallback to argmax.
    If multiple classes pass, choose the class with the largest margin
    probability - threshold.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if thresholds.shape[0] != probabilities.shape[1]:
        raise ValueError("threshold count must match number of classes")

    margins = probabilities - thresholds.reshape(1, -1)
    passes = margins >= 0.0
    threshold_preds = margins.argmax(axis=1)
    argmax_preds = probabilities.argmax(axis=1)
    return np.where(passes.any(axis=1), threshold_preds, argmax_preds)


def macro_fbeta_score(y_true, y_pred, beta=2.0, num_classes=4, abnormal_only=True, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    class_ids = range(1, num_classes) if abnormal_only else range(num_classes)
    beta_sq = beta ** 2
    scores = []

    for class_id in class_ids:
        tp = np.logical_and(y_true == class_id, y_pred == class_id).sum()
        fp = np.logical_and(y_true != class_id, y_pred == class_id).sum()
        fn = np.logical_and(y_true == class_id, y_pred != class_id).sum()

        numerator = (1.0 + beta_sq) * tp
        denominator = numerator + beta_sq * fn + fp
        scores.append(numerator / (denominator + eps))

    return float(np.mean(scores)) if scores else 0.0


def find_best_thresholds(
    probabilities,
    y_true,
    grid_values=None,
    objective="icbhi",
    beta=2.0,
    num_classes=4,
):
    """
    Grid-search per-class thresholds.

    objective:
        icbhi       -> maximize ICBHI score
        sensitivity -> maximize sensitivity
        fbeta       -> maximize abnormal macro F-beta
    """
    if grid_values is None:
        grid_values = np.arange(0.1, 1.0, 0.1)

    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    grid_values = [float(value) for value in grid_values]

    best_thresholds = np.full(num_classes, 0.5, dtype=np.float64)
    best_metrics = None
    best_value = -np.inf

    for candidate in itertools.product(grid_values, repeat=num_classes):
        preds = predict_with_thresholds(probabilities, candidate)
        metrics = compute_icbhi_metrics(y_true, preds, num_classes=num_classes)

        if objective == "icbhi":
            value = metrics["score"]
        elif objective == "sensitivity":
            value = metrics["sensitivity"]
        elif objective == "fbeta":
            value = macro_fbeta_score(
                y_true,
                preds,
                beta=beta,
                num_classes=num_classes,
                abnormal_only=True,
            )
        else:
            raise ValueError(f"Unknown threshold objective: {objective}")

        if value > best_value:
            best_value = value
            best_thresholds = np.asarray(candidate, dtype=np.float64)
            best_metrics = metrics

    return best_thresholds, best_metrics


def parse_thresholds(text):
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    return np.asarray(values, dtype=np.float64)


def save_thresholds(path, thresholds, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"thresholds": [float(value) for value in thresholds]}
    if metrics is not None:
        payload["metrics"] = {
            "sensitivity": float(metrics["sensitivity"]),
            "specificity": float(metrics["specificity"]),
            "score": float(metrics["score"]),
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_thresholds(path):
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = payload["thresholds"]

    return np.asarray(payload, dtype=np.float64)


def write_prediction_reports(output_dir, targets, predictions, probabilities, device_ids=None):
    """
    Save prediction-level CSV reports for error analysis.

    The preprocessed .npz does not store original recording names, so reports use
    test sample indices and device ids.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if device_ids is None:
        device_ids = np.full(len(targets), -1, dtype=np.int64)
    else:
        device_ids = np.asarray(device_ids, dtype=np.int64)

    header = [
        "sample_index",
        "device_id",
        "true_label",
        "true_class",
        "pred_label",
        "pred_class",
        "prob_normal",
        "prob_crackle",
        "prob_wheeze",
        "prob_both",
    ]

    rows = []
    false_negative_rows = []
    normal_as_abnormal_rows = []

    for idx, (target, pred, probs, device_id) in enumerate(
        zip(targets, predictions, probabilities, device_ids)
    ):
        row = [
            idx,
            int(device_id),
            int(target),
            CLASS_NAMES[int(target)],
            int(pred),
            CLASS_NAMES[int(pred)],
            float(probs[0]),
            float(probs[1]),
            float(probs[2]),
            float(probs[3]),
        ]
        rows.append(row)

        if int(target) != 0 and int(pred) == 0:
            false_negative_rows.append(row)
        if int(target) == 0 and int(pred) != 0:
            normal_as_abnormal_rows.append(row)

    outputs = {
        "predictions.csv": rows,
        "false_negatives_abnormal_as_normal.csv": false_negative_rows,
        "normal_as_abnormal.csv": normal_as_abnormal_rows,
    }

    for filename, file_rows in outputs.items():
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(file_rows)

    return {name: str(output_dir / name) for name in outputs}
