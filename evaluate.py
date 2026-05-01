import argparse
import csv
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor

from src.dataset import ASTDataset
from src.model import CustomAST
from src.utils import (
    CLASS_NAMES,
    compute_icbhi_metrics,
    find_best_thresholds,
    load_thresholds,
    parse_thresholds,
    predict_with_thresholds,
    save_thresholds,
    write_prediction_reports,
)


NUM_CLASSES = 4


def threshold_grid(args):
    return np.arange(args.threshold_min, args.threshold_max + 1e-8, args.threshold_step)


def choose_predictions(probabilities, targets, args):
    thresholds = None
    tuned_metrics = None

    if args.thresholds:
        thresholds = parse_thresholds(args.thresholds)
        print(f"Using manual thresholds: {thresholds}")

    if args.thresholds_path:
        thresholds = load_thresholds(args.thresholds_path)
        print(f"Using thresholds from {args.thresholds_path}: {thresholds}")

    if args.tune_thresholds:
        print(
            "Threshold tuning is running on this evaluation set. "
            "For final papers/reports, prefer a validation set if your team creates one."
        )
        thresholds, tuned_metrics = find_best_thresholds(
            probabilities,
            targets,
            grid_values=threshold_grid(args),
            objective=args.threshold_objective,
            beta=args.fbeta_beta,
            num_classes=NUM_CLASSES,
        )
        print(f"Best thresholds: {thresholds}")

    if thresholds is None:
        return probabilities.argmax(axis=1), None, tuned_metrics

    return predict_with_thresholds(probabilities, thresholds), thresholds, tuned_metrics


def save_metrics_csv(path, metrics):
    rows = [
        ["sensitivity", metrics["sensitivity"]],
        ["specificity", metrics["specificity"]],
        ["icbhi_score", metrics["score"]],
        ["normal_as_abnormal", metrics["normal_as_abnormal"]],
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        rows.append([f"recall_{class_name.lower()}", metrics["per_class_recall"][class_id]])
    for class_id in range(1, NUM_CLASSES):
        rows.append([
            f"false_negative_{CLASS_NAMES[class_id].lower()}_as_normal",
            metrics["false_negatives_as_normal"][class_id],
        ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def evaluate(args):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    data = np.load(args.data_path)
    X_test, y_test, d_test = data["X_test"], data["y_test"], data["device_test"]

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    test_dataset = ASTDataset(X_test, y_test, d_test, processor, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Loading model: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model = CustomAST(num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Evaluating...")
    all_probs = []
    all_targets = []
    all_device_ids = []

    with torch.no_grad():
        for batch_idx, (inputs, labels, device_ids) in enumerate(test_loader):
            if args.limit_batches and args.limit_batches > 0 and batch_idx >= args.limit_batches:
                break

            inputs = inputs.to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(inputs)
            else:
                logits = model(inputs)

            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_device_ids.extend(np.asarray(device_ids))

    probabilities = np.concatenate(all_probs, axis=0)
    targets = np.asarray(all_targets, dtype=np.int64)
    device_ids = np.asarray(all_device_ids, dtype=np.int64)

    predictions, thresholds, tuned_metrics = choose_predictions(probabilities, targets, args)
    metrics = compute_icbhi_metrics(targets, predictions, num_classes=NUM_CLASSES)
    cm = metrics["confusion_matrix"]

    print("\nMetrics:")
    print(f"  Sensitivity (Se): {metrics['sensitivity']:.4f} ({metrics['sensitivity'] * 100:.2f}%)")
    print(f"  Specificity (Sp): {metrics['specificity']:.4f} ({metrics['specificity'] * 100:.2f}%)")
    print(f"  ICBHI Score:      {metrics['score']:.4f} ({metrics['score'] * 100:.2f}%)")

    print("\nRecall per class:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {metrics['per_class_recall'][class_id]:.4f}")

    print("\nFalse-negative analysis:")
    for class_id in range(1, NUM_CLASSES):
        print(
            f"  {CLASS_NAMES[class_id]} predicted as Normal: "
            f"{metrics['false_negatives_as_normal'][class_id]}"
        )
    print(f"  Normal predicted as abnormal: {metrics['normal_as_abnormal']}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        report_paths = write_prediction_reports(
            args.output_dir,
            targets,
            predictions,
            probabilities,
            device_ids=device_ids,
        )
        for path in report_paths.values():
            print(f"Saved report: {path}")

        metrics_path = os.path.join(args.output_dir, "metrics_summary.csv")
        save_metrics_csv(metrics_path, metrics)
        print(f"Saved metrics: {metrics_path}")

        if thresholds is not None and args.save_thresholds_path:
            save_thresholds(args.save_thresholds_path, thresholds, tuned_metrics or metrics)
            print(f"Saved thresholds: {args.save_thresholds_path}")

        plt.figure(figsize=(8, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"label": "Number of Samples"},
        )

        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)

        metrics_text = (
            f"Sensitivity: {metrics['sensitivity'] * 100:.2f}%  |  "
            f"Specificity: {metrics['specificity'] * 100:.2f}%  |  "
            f"Score: {metrics['score'] * 100:.2f}%"
        )
        plt.figtext(
            0.5,
            0.02,
            metrics_text,
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", boxstyle="round,pad=0.5"),
        )
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        fig_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(fig_path, dpi=600, bbox_inches="tight")
        print(f"Saved confusion matrix: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AST model for ICBHI")
    parser.add_argument("--data_path", type=str, default="./data/icbhi_preprocessed.npz")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_batches", type=int, default=0)

    parser.add_argument("--thresholds", type=str, default=None, help="Manual thresholds, e.g. 0.5,0.4,0.4,0.3")
    parser.add_argument("--thresholds_path", type=str, default=None)
    parser.add_argument("--save_thresholds_path", type=str, default=None)
    parser.add_argument("--tune_thresholds", action="store_true")
    parser.add_argument("--threshold_objective", choices=["icbhi", "sensitivity", "fbeta"], default="icbhi")
    parser.add_argument("--threshold_min", type=float, default=0.1)
    parser.add_argument("--threshold_max", type=float, default=0.9)
    parser.add_argument("--threshold_step", type=float, default=0.1)
    parser.add_argument("--fbeta_beta", type=float, default=2.0)

    args = parser.parse_args()
    evaluate(args)
