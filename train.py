import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import ASTFeatureExtractor

from src.dataset import ASTDataset
from src.losses import (
    FBetaLoss,
    FocalLoss,
    WeightedCrossEntropyLoss,
    load_or_compute_class_weights,
)
from src.model import CustomAST
from src.sam import SAM
from src.utils import CLASS_NAMES, compute_icbhi_metrics


NUM_CLASSES = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sampler(y_train, sampler_name):
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)

    if sampler_name == "none":
        return None

    if sampler_name == "weighted":
        sample_weights = np.array([1.0 / max(counts[label], 1.0) for label in y_train])
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    if sampler_name == "minority":
        # Focus on the rare ICBHI classes: Wheeze=2 and Both=3.
        sample_weights = np.ones(len(y_train), dtype=np.float64)
        max_count = max(float(counts.max()), 1.0)
        for label in (2, 3):
            sample_weights[y_train == label] = max_count / max(counts[label], 1.0)
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    raise ValueError(f"Unknown sampler: {sampler_name}")


def build_criterion(args, npz_data, y_train, device):
    class_weights = load_or_compute_class_weights(
        npz_data,
        y_train,
        num_classes=NUM_CLASSES,
        source=args.class_weight_source,
    ).to(device)

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss == "weighted_ce":
        criterion = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )
    elif args.loss == "focal":
        alpha = class_weights if args.focal_alpha == "class_weights" else None
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            alpha=alpha,
            label_smoothing=args.label_smoothing,
        )
    elif args.loss == "fbeta":
        criterion = FBetaLoss(
            beta=args.fbeta_beta,
            num_classes=NUM_CLASSES,
            abnormal_only=args.fbeta_abnormal_only,
        )
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    print(f"Class counts: {np.bincount(y_train, minlength=NUM_CLASSES)}")
    print(f"Class weights: {class_weights.detach().cpu().numpy()}")
    print(f"Loss: {args.loss}")
    if args.loss == "focal":
        print(f"Focal gamma: {args.focal_gamma} | alpha: {args.focal_alpha}")
    if args.loss == "fbeta":
        print(f"F-beta beta: {args.fbeta_beta} | abnormal_only: {args.fbeta_abnormal_only}")

    return criterion.to(device)


def limited_iterator(loader, limit_batches):
    if limit_batches is None or limit_batches <= 0:
        yield from loader
        return

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= limit_batches:
            break
        yield batch


def evaluate_on_loader(model, loader, device, limit_batches=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels, _ in limited_iterator(loader, limit_batches):
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return compute_icbhi_metrics(all_labels, all_preds, num_classes=NUM_CLASSES)


def checkpoint_path(args):
    filename = args.run_name if args.run_name.endswith(".pth") else f"{args.run_name}.pth"
    return os.path.join(args.checkpoint_dir, filename)


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading data: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}. Run preprocess.py first.")

    data = np.load(args.data_path)
    X_train, y_train, d_train = data["X_train"], data["y_train"], data["device_train"]
    X_test, y_test, d_test = data["X_test"], data["y_test"], data["device_test"]

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    sampler = build_sampler(y_train, args.sampler)
    print(f"Sampler: {args.sampler}")

    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True),
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("Preparing model")
    model = CustomAST(num_classes=NUM_CLASSES).to(device)

    optimizer = SAM(
        model.parameters(),
        torch.optim.AdamW,
        lr=args.lr,
        rho=args.sam_rho,
        weight_decay=args.weight_decay,
    )
    criterion = build_criterion(args, data, y_train, device)

    save_path = checkpoint_path(args)
    best_score = -1.0

    print("Training begins")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        if args.limit_train_batches and args.limit_train_batches > 0:
            total_batches = min(len(train_loader), args.limit_train_batches)
        else:
            total_batches = len(train_loader)

        progress_bar = tqdm(
            limited_iterator(train_loader, args.limit_train_batches),
            total=total_batches,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
        )

        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            second_loss = criterion(model(inputs), labels)
            second_loss.backward()
            optimizer.second_step(zero_grad=True)

            running_loss += float(loss.item())
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = evaluate_on_loader(
            model,
            test_loader,
            device,
            limit_batches=args.limit_eval_batches,
        )
        avg_loss = running_loss / max(num_batches, 1)
        score = metrics["score"]

        print(
            f"Epoch {epoch + 1}: loss={avg_loss:.4f} | "
            f"Score={score:.4f} | Se={metrics['sensitivity']:.4f} | "
            f"Sp={metrics['specificity']:.4f}"
        )
        for class_id, class_name in enumerate(CLASS_NAMES):
            print(f"  Recall {class_name}: {metrics['per_class_recall'][class_id]:.4f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best checkpoint: {save_path}")

    print(f"\nBest ICBHI score: {best_score:.4f}")
    print(f"Best checkpoint: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AST+SAM with imbalance strategies for ICBHI")
    parser.add_argument("--data_path", type=str, default="./data/icbhi_preprocessed.npz")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default="best_model")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--sampler", choices=["weighted", "minority", "none"], default="weighted")
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal", "fbeta"], default="ce")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--class_weight_source", choices=["npz", "train"], default="npz")

    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", choices=["none", "class_weights"], default="none")

    parser.add_argument("--fbeta_beta", type=float, default=2.0)
    parser.add_argument("--fbeta_abnormal_only", action="store_true")

    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)

    args = parser.parse_args()
    train(args)
