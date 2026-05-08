import torch
import torch.nn as nn
import torch.nn.functional as F


class FBetaLoss(nn.Module):
    """
    Differentiable macro F-beta loss.

    beta=2 gives recall more importance than precision, which is useful when
    the goal is to reduce false negatives in abnormal respiratory cycles.
    """

    def __init__(self, beta=2.0, num_classes=4, abnormal_only=False, eps=1e-7):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")

        self.beta = float(beta)
        self.num_classes = int(num_classes)
        self.abnormal_only = bool(abnormal_only)
        self.eps = float(eps)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        tp = (probs * target_one_hot).sum(dim=0)
        fp = (probs * (1.0 - target_one_hot)).sum(dim=0)
        fn = ((1.0 - probs) * target_one_hot).sum(dim=0)

        beta_sq = self.beta ** 2
        fbeta = ((1.0 + beta_sq) * tp + self.eps) / (
            (1.0 + beta_sq) * tp + beta_sq * fn + fp + self.eps
        )

        if self.abnormal_only and self.num_classes > 1:
            fbeta = fbeta[1:]

        return 1.0 - fbeta.mean()
