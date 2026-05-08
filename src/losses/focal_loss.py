import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss for single-label classification.

    Formula:
        FL(p_t) = alpha_t * (1 - p_t)^gamma * CE

    gamma values to test for this project: 0.5, 1.0, 2.0.
    alpha can be None or a tensor of per-class weights.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be mean, sum, or none")

        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
