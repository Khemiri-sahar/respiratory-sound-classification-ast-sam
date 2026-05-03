import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import ASTFeatureExtractor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import os
import argparse
import gc
import json
from pathlib import Path

from src.dataset import ASTDataset
from src.model import CustomAST

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def evaluate_model(model, test_loader, device, use_amp=True):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs = inputs.to(device)
            
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(inputs)
            else:
                logits = model(inputs)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    return np.array(all_preds), np.array(all_targets)


def calculate_metrics(cm, class_names):
    se_numerator = np.sum(cm[1:, 1:])
    se_denominator = np.sum(cm[1:, :])
    se = se_numerator / se_denominator if se_denominator > 0 else 0.0
    
    sp_numerator = cm[0, 0]
    sp_denominator = np.sum(cm[0, :])
    sp = sp_numerator / sp_denominator if sp_denominator > 0 else 0.0
    
    score = (se + sp) / 2.0
    
    precision, recall, f1, support = precision_recall_fscore_support(
        np.repeat(np.arange(4), np.diag(cm)), 
        np.repeat(np.arange(4), np.sum(cm, axis=1)),
        average=None, 
        zero_division=0
    )
    
    # Alternative: direct calculation from confusion matrix
    class_metrics = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        class_metrics[name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'support': tp + fn
        }
    
    return se, sp, score, class_metrics


def plot_confusion_matrix(cm, class_names, output_path, metrics_dict):
    """Plot and save confusion matrix."""
    se = metrics_dict['se']
    sp = metrics_dict['sp']
    score = metrics_dict['score']
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 12, "weight": "bold"},
                     cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - ICBHI 2017', fontsize=14, fontweight='bold', pad=15)
    
    # Add metrics text
    metrics_text = f"Sensitivity (Se): {se*100:.2f}%  |  Specificity (Sp): {sp*100:.2f}%  |  Score: {score*100:.2f}%"
    plt.figtext(0.5, 0.02, metrics_text, wrap=True, horizontalalignment='center', 
                fontsize=11, fontweight='bold', bbox=dict(facecolor='lightblue', alpha=0.8, 
                edgecolor='black', boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_per_class_metrics(class_metrics, class_names, output_path):
    """Plot per-class metrics."""
    metrics_to_plot = ['sensitivity', 'specificity', 'precision', 'f1']
    x = np.arange(len(class_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [class_metrics[name][metric] for name in class_names]
        ax.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def plot_training_history(history_path, output_path):
    """Plot training history from JSON file."""
    if not os.path.exists(history_path):
        print(f"   ⚠️  Training history not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Sensitivity curve
    axes[0, 1].plot(epochs, np.array(history['val_se'])*100, 'g-o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Sensitivity (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Validation Sensitivity (Se)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Specificity curve
    axes[1, 0].plot(epochs, np.array(history['val_sp'])*100, 'r-o', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Specificity (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Validation Specificity (Sp)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # ICBHI Score curve
    axes[1, 1].plot(epochs, np.array(history['val_score'])*100, 'm-o', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('ICBHI Score (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('ICBHI Score = (Se + Sp) / 2', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # Add config text
    config = history.get('config', {})
    config_text = f"Optimizer: {'SAM' if config.get('use_sam') else 'AdamW'} | "
    config_text += f"FP16: {config.get('use_amp')} | LR: {config.get('lr')} | BS: {config.get('batch_size')}"
    fig.suptitle(f'Training History - {config_text}', fontsize=13, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def evaluate(args):
    """Main evaluation function."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASS_NAMES = ['Normal', 'Crackle', 'Wheeze', 'Both']
    
    print(f"\n{'='*80}")
    print(f"📊 EVALUATION - ICBHI 2017 Respiratory Sound Classification")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Mixed-Precision (FP16): {args.use_amp and DEVICE.type == 'cuda'}\n")
    
    # Load data
    print(f"📥 Loading data: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    data = np.load(args.data_path)
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']
    print(f"   ✓ Test samples: {len(X_test)}")

    processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    test_dataset = ASTDataset(X_test, y_test, d_test, processor, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model
    print(f"\n📦 Loading model: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}. Run train.py first.")

    model = CustomAST(num_classes=4).to(DEVICE)
    try:
        state_dict = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"   ✓ Model weights loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return

    # Evaluate
    print(f"\n🔍 Running evaluation...")
    all_preds, all_targets = evaluate_model(model, test_loader, DEVICE, args.use_amp)

    # Calculate metrics
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])
    se, sp, score, class_metrics = calculate_metrics(cm, CLASS_NAMES)

    # Print results
    print(f"\n{'='*80}")
    print(f"📊 RESULTS")
    print(f"{'='*80}")
    print(f"\n🎯 ICBHI Metrics:")
    print(f"   Sensitivity (Se):  {se*100:6.2f}%")
    print(f"   Specificity (Sp):  {sp*100:6.2f}%")
    print(f"   ICBHI Score:       {score*100:6.2f}%")
    
    print(f"\n📈 Per-Class Metrics:")
    print(f"   {'Class':<10} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<12} {'F1':<10}")
    print(f"   {'-'*56}")
    for name in CLASS_NAMES:
        metrics = class_metrics[name]
        print(f"   {name:<10} {metrics['sensitivity']*100:>10.2f}% {metrics['specificity']*100:>10.2f}% "
              f"{metrics['precision']*100:>10.2f}% {metrics['f1']*100:>8.2f}%")
    
    print(f"\n\nConfusion Matrix:")
    print(f"   {' '*15}{'Predicted':<40}")
    print(f"   {' '*10}{CLASS_NAMES[0]:>8} {CLASS_NAMES[1]:>8} {CLASS_NAMES[2]:>8} {CLASS_NAMES[3]:>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"   {name:<10} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8} {cm[i, 3]:>8}")

    # Save visualizations
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n💾 Saving visualizations to {args.output_dir}...")
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, CLASS_NAMES, cm_path, 
                            {'se': se, 'sp': sp, 'score': score})
        
        # Per-class metrics
        metrics_path = os.path.join(args.output_dir, "per_class_metrics.png")
        plot_per_class_metrics(class_metrics, CLASS_NAMES, metrics_path)
        
        # Training history
        checkpoint_dir = os.path.dirname(args.model_path)
        history_path = os.path.join(checkpoint_dir, "training_history.json")
        if os.path.exists(history_path):
            history_plot = os.path.join(args.output_dir, "training_history.png")
            plot_training_history(history_path, history_plot)
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of AST+SAM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data_path", type=str, default="./data/icbhi_ast_16k_8s_metadata.npz", 
                        help="Path to preprocessed .npz file")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth", 
                        help="Path to trained model (.pth)")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="Directory to save evaluation plots")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for evaluation")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use mixed-precision (FP16) for evaluation")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false",
                        help="Disable mixed-precision")
    
    args = parser.parse_args()
    evaluate(args)
