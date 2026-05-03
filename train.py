"""
Training script for AST + SAM optimization on ICBHI 2017 respiratory sound classification.

Features:
  - AST pre-trained on AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593)
  - SAM (Sharpness-Aware Minimization) optimizer integration
  - Mixed-precision (FP16) training with torch.cuda.amp
  - Ablation support: AST+SAM vs AST+AdamW
  - Per-class sensitivity analysis
  - Training history and metrics logging
  - Weighted sampling for class imbalance handling

Official Split: 60% train, 40% test
Training config: 20 epochs, batch size 8, lr=1e-5, rho=0.05 (SAM)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime

from src.dataset import ASTDataset
from src.model import CustomAST
from src.sam import SAM


def calculate_metrics(cm):
    """Calculate Sensitivity (Se), Specificity (Sp), and ICBHI Score from confusion matrix.
    
    ICBHI scoring:
      - Se = TP(abnormal) / (TP(abnormal) + FN(abnormal))
      - Sp = TN(normal) / (TN(normal) + FP(normal))
      - Score = (Se + Sp) / 2
    """
    # Se = (TP_crackle + TP_wheeze + TP_both) / (all abnormal)
    se_numerator = np.sum(cm[1:, 1:])
    se_denominator = np.sum(cm[1:, :])
    se = se_numerator / se_denominator if se_denominator > 0 else 0.0
    
    # Sp = TN_normal / (TN_normal + FP_normal)
    sp_numerator = cm[0, 0]
    sp_denominator = np.sum(cm[0, :])
    sp = sp_numerator / sp_denominator if sp_denominator > 0 else 0.0
    
    # ICBHI Score = (Se + Sp) / 2
    score = (se + sp) / 2.0
    
    # Per-class sensitivity
    class_sensitivities = {}
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    for i in range(4):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        class_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_sensitivities[class_names[i]] = class_sens
    
    return se, sp, score, class_sensitivities


def train(args):
    """Train AST model with optional SAM optimizer and mixed-precision."""
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"🚀 AST + SAM TRAINING - ICBHI 2017 Respiratory Sound Classification")
    print(f"{'='*80}")
    print(f"⚙️  Device: {DEVICE}")
    print(f"   Compute: GPU with Mixed-Precision (FP16)" if args.use_amp and DEVICE.type == 'cuda' 
          else f"   Compute: CPU (FP32)")
    print(f"   Optimizer: {'SAM (Sharpness-Aware Minimization, rho={:.4f})'.format(args.rho) if args.use_sam else 'AdamW'}")
    print(f"   Config: {args.epochs} epochs | batch_size={args.batch_size} | lr={args.lr}")
    print(f"   Model: MIT/ast-finetuned-audioset-10-10-0.4593 (pretrained on AudioSet)")
    print(f"{'='*80}\n")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # ========== Load Data ==========
    print(f"📥 Loading preprocessed data: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}\nRun preprocess.py first.")

    data = np.load(args.data_path)
    X_train, y_train, d_train = data['X_train'], data['y_train'], data['device_train']
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']
    
    print(f"   ✓ Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"   ✓ Train class distribution: {np.bincount(y_train)}")

    # ========== Setup DataLoaders ==========
    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    # Weighted sampler for class imbalance
    counts = np.bincount(y_train)
    weights = [1.0/counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, processor, train=True), 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=0
    )
    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, processor, train=False), 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )

    # ========== Initialize Model ==========
    print("\n🧠 Initializing model...")
    model = CustomAST(num_classes=4).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    
    # ========== Resume Training Setup ==========
    resume_from_checkpoint = None
    start_epoch = 0
    total_epochs = args.epochs
    
    if args.resume_epochs > 0:
        # Auto-detect checkpoint path if not provided
        if args.resume_from is None:
            args.resume_from = os.path.join(args.checkpoint_dir, "best_model.pth")
        
        if os.path.exists(args.resume_from):
            print(f"\n🔄 RESUME MODE: Loading checkpoint from {args.resume_from}")
            model.load_state_dict(torch.load(args.resume_from, map_location=DEVICE))
            print(f"   ✓ Model loaded successfully")
            total_epochs = args.epochs + args.resume_epochs
            print(f"   ✓ Original epochs: {args.epochs} → Extended epochs: {total_epochs}")
            
            # Try to load training history to continue from best_score
            history_path = os.path.join(args.checkpoint_dir, "training_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    old_history = json.load(f)
                    start_epoch = len(old_history.get('epoch', []))
                    resume_from_checkpoint = old_history
                    print(f"   ✓ History loaded: {start_epoch} epochs already trained")
        else:
            print(f"⚠️  Checkpoint not found at {args.resume_from}. Starting fresh training.")
            args.resume_epochs = 0
    
    # ========== Setup Optimizer ==========
    if args.use_sam:
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            model.parameters(), 
            base_optimizer, 
            lr=args.lr, 
            rho=args.rho,
            weight_decay=1e-4
        )
        optim_name = f"SAM (rho={args.rho:.4f})"
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        optim_name = "AdamW"
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and DEVICE.type == 'cuda' else None
    
    print(f"\n📊 Optimizer: {optim_name}")
    print(f"   Loss function: CrossEntropyLoss (label_smoothing=0.1)")
    print(f"   LR scheduler: None")
    
    # ========== Training History ==========
    if resume_from_checkpoint:
        history = resume_from_checkpoint
    else:
        history = {
            'config': {
                'use_sam': args.use_sam,
                'rho': args.rho if args.use_sam else None,
                'use_amp': args.use_amp,
                'lr': args.lr,
                'batch_size': args.batch_size,
                'epochs': total_epochs
            },
            'epoch': [], 'train_loss': [], 'val_se': [], 'val_sp': [], 'val_score': [],
            'class_sensitivities': []
        }
    
    best_score = max(history.get('val_score', [0])) if history.get('val_score') else 0.0
    best_epoch = history.get('epoch', [0])[-1] if history.get('epoch') else 0

    print(f"\n{'='*80}")
    print(f"Starting training with {optim_name}...")
    if args.resume_epochs > 0:
        print(f"🔄 Resuming: epochs {start_epoch+1}-{total_epochs} (adding {args.resume_epochs} epochs)")
    print(f"{'='*80}\n")
    
    # ========== Training Loop ==========
    for epoch in range(start_epoch, total_epochs):
        # -------- Training phase --------
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"[{epoch+1:2d}/{total_epochs}] TRAIN", leave=False)
        
        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if args.use_amp and DEVICE.type == 'cuda':
                # ===== Mixed-precision training (FP16) =====
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                
                scaler.scale(loss).backward()
                
                if args.use_sam:
                    # SAM first step (don't unscale yet)
                    optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass at perturbed weights
                    with torch.cuda.amp.autocast():
                        criterion(model(inputs), labels).backward()
                    
                    # Now unscale and apply second step
                    scaler.unscale_(optimizer)
                    optimizer.second_step(zero_grad=True)
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # ===== Standard precision training (FP32) =====
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                
                if args.use_sam:
                    optimizer.first_step(zero_grad=True)
                    criterion(model(inputs), labels).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_loss / num_batches

        # -------- Validation phase --------
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(test_loader, desc=f"[{epoch+1:2d}/{total_epochs}] EVAL", leave=False):
                inputs = inputs.to(DEVICE)
                
                if args.use_amp and DEVICE.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(inputs)
                else:
                    logits = model(inputs)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
        se, sp, score, class_sens = calculate_metrics(cm)

        # Store history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['val_se'].append(float(se))
        history['val_sp'].append(float(sp))
        history['val_score'].append(float(score))
        history['class_sensitivities'].append({k: float(v) for k, v in class_sens.items()})

        # Print metrics
        print(f"\n[Epoch {epoch+1:2d}] Loss: {avg_train_loss:.4f} | Se: {se*100:5.2f}% | Sp: {sp*100:5.2f}% | Score: {score*100:5.2f}%")
        print(f"             Per-class Se → Normal: {class_sens['Normal']*100:5.2f}% | "
              f"Crackle: {class_sens['Crackle']*100:5.2f}% | "
              f"Wheeze: {class_sens['Wheeze']*100:5.2f}% | "
              f"Both: {class_sens['Both']*100:5.2f}%")

        # Save best model
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"             ✅ Best model saved (Score: {score*100:.2f}%)")

    # ========== Final Summary ==========
    print(f"\n{'='*80}")
    print(f"🏆 TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Score: {best_score*100:.2f}%")
    best_metrics = history['class_sensitivities'][best_epoch - 1]
    print(f"  - Sensitivity (Se): {history['val_se'][best_epoch-1]*100:.2f}%")
    print(f"  - Specificity (Sp): {history['val_sp'][best_epoch-1]*100:.2f}%")
    print(f"  - Per-class sensitivity:")
    print(f"      • Normal:  {best_metrics['Normal']*100:.2f}%")
    print(f"      • Crackle: {best_metrics['Crackle']*100:.2f}%")
    print(f"      • Wheeze:  {best_metrics['Wheeze']*100:.2f}%")
    print(f"      • Both:    {best_metrics['Both']*100:.2f}%")
    print(f"\n📁 Model saved: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    
    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📊 History saved: {history_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AST+SAM for respiratory sound classification (ICBHI 2017)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data/icbhi_ast_16k_8s_metadata.npz", 
                        help="Path to preprocessed .npz file from preprocess.py")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", 
                        help="Directory to save model checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs (recommended: 20)")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training (recommended: 8)")
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="Learning rate for optimizer (recommended: 1e-5)")
    
    # Ablation study: SAM vs AdamW
    parser.add_argument("--use_sam", action="store_true", default=True,
                        help="Use SAM optimizer (default: True)")
    parser.add_argument("--no_sam", dest="use_sam", action="store_false",
                        help="Disable SAM, use AdamW only (for ablation study)")
    parser.add_argument("--rho", type=float, default=0.05,
                        help="SAM rho parameter - neighborhood size (recommended: 0.05)")
    
    # Mixed-precision training
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use Automatic Mixed Precision (FP16) for GPU training")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false",
                        help="Disable mixed-precision, use FP32 instead")
    
    # Resume training
    parser.add_argument("--resume_epochs", type=int, default=0,
                        help="Resume training: number of additional epochs to add (e.g., 5 to add 5 more epochs)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from (default: ./checkpoints/ast_adamw_fp16_extended/best_model.pth)")
    
    args = parser.parse_args()
    train(args)
