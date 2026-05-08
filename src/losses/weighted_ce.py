import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(labels, num_classes=4, eps=1e-8):
    """
    Compute inverse-frequency class weights.

    Formula required by the project:
        w_c = N_total / (num_classes * N_c)
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = float(counts.sum())

    weights = np.zeros(num_classes, dtype=np.float32)
    valid = counts > 0
    weights[valid] = total / (num_classes * counts[valid] + eps)
    return torch.tensor(weights, dtype=torch.float32)


def load_or_compute_class_weights(npz_data, y_train, num_classes=4, source="npz"):
    """
    Return class weights from preprocess.py output or compute them from y_train.

    source="npz" keeps the weights produced by preprocessing.
    source="train" recomputes weights from the current y_train array.
    """
    if source == "npz" and "class_weights" in npz_data.files:
        return torch.tensor(npz_data["class_weights"], dtype=torch.float32)

    if source == "train" or source == "npz":
        return compute_class_weights(y_train, num_classes=num_classes)

    raise ValueError(f"Unknown class weight source: {source}")


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-Entropy with per-class weights for imbalanced classification."""

    def __init__(self, class_weights, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits, targets):
        return self.loss(logits, targets)
