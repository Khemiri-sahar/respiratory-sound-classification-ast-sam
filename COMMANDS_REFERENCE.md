# 🎯 Quick Command Reference - AST + SAM Training Pipeline

## One-Line Setup

```bash
# Full setup from scratch
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python preprocess.py && python train.py && python evaluate.py
```

---

## Phase 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Verification**:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## Phase 2: Data Preparation

### Download (Manual Steps)
1. Visit: https://bhichallenge.med.auth.gr/
2. Download `ICBHI_final_database.zip` (~5GB)
3. Download `ICBHI_challenge_train_test.txt`
4. Extract to `./data/`

### Preprocess
```bash
# Standard preprocessing (with augmentation)
python preprocess.py

# Without augmentation (faster testing)
python preprocess.py --no_augment

# Custom configuration
python preprocess.py \
    --duration 6 \
    --n_mels 64 \
    --hop_length 1024
```

**Check**: `ls -lh data/icbhi_ast_16k_8s_metadata.npz` (should be ~2-3GB)

---

## Phase 3: Training

### Default (AST + SAM + FP16)
```bash
python train.py
```

### All Common Variants

#### Option 1: AST + SAM (Proposed - Recommended)
```bash
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --use_sam \
    --rho 0.05 \
    --use_amp \
    --checkpoint_dir ./checkpoints/ast_sam_fp16
```

#### Option 2: AST + AdamW (Baseline)
```bash
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --no_sam \
    --use_amp \
    --checkpoint_dir ./checkpoints/ast_adamw_fp16
```

#### Option 3: AST + SAM (FP32 - CPU Compatible)
```bash
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --use_sam \
    --rho 0.05 \
    --no_amp \
    --checkpoint_dir ./checkpoints/ast_sam_fp32
```

#### Option 4: Quick Test (1 Epoch)
```bash
python train.py --epochs 1 --batch_size 4
```

#### Option 5: Large-Scale (More epochs)
```bash
python train.py \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-5 \
    --use_sam \
    --rho 0.05 \
    --use_amp
```

---

## Phase 4: Ablation Studies

### Run All Ablations
```bash
# AST + SAM with different rho values
python train.py --rho 0.01 --checkpoint_dir ./checkpoints/ast_sam_rho_001
python train.py --rho 0.05 --checkpoint_dir ./checkpoints/ast_sam_rho_005  # default
python train.py --rho 0.1 --checkpoint_dir ./checkpoints/ast_sam_rho_010

# AST + AdamW
python train.py --no_sam --checkpoint_dir ./checkpoints/ast_adamw

# AST + SAM without FP16
python train.py --use_sam --rho 0.05 --no_amp --checkpoint_dir ./checkpoints/ast_sam_fp32
```

### Automated Ablation Loop
```bash
for rho in 0.01 0.05 0.1; do
    echo "=== Training with rho=$rho ==="
    python train.py \
        --use_sam \
        --rho $rho \
        --use_amp \
        --epochs 20 \
        --checkpoint_dir ./checkpoints/ast_sam_rho_${rho}
done
```

---

## Phase 5: Evaluation

### Single Model
```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path ./data/icbhi_ast_16k_8s_metadata.npz \
    --output_dir ./results
```

### Batch Evaluation (All Ablations)
```bash
mkdir -p results

for model_dir in checkpoints/ast_sam* checkpoints/ast_adamw*; do
    model_name=$(basename $model_dir)
    echo "Evaluating $model_name..."
    
    python evaluate.py \
        --model_path $model_dir/best_model.pth \
        --output_dir ./results/$model_name
done
```

### Quick Evaluation (Lower batch size for memory)
```bash
python evaluate.py --batch_size 8
```

---

## Phase 6: Analysis & Comparison

### View Training History
```bash
# Print best epoch metrics
python -c "import json; h=json.load(open('./checkpoints/training_history.json')); 
best_idx = h['val_score'].index(max(h['val_score'])); 
print(f\"Best Epoch: {h['epoch'][best_idx]}\")
print(f\"Score: {h['val_score'][best_idx]*100:.2f}%\")"
```

### Compare Multiple Models
```bash
python -c "
import json
import os

models = ['ast_sam_fp16', 'ast_adamw_fp16', 'ast_sam_fp32']
print(f'{\"Model\":<20} {\"Best Se\":>10} {\"Best Sp\":>10} {\"Best Score\":>10}')
print('-' * 50)

for model in models:
    history_file = f'./checkpoints/{model}/training_history.json'
    if os.path.exists(history_file):
        h = json.load(open(history_file))
        best_idx = h['val_score'].index(max(h['val_score']))
        se = h['val_se'][best_idx] * 100
        sp = h['val_sp'][best_idx] * 100
        score = h['val_score'][best_idx] * 100
        print(f'{model:<20} {se:>9.2f}% {sp:>9.2f}% {score:>9.2f}%')
"
```

---

## 🔍 Monitoring & Debugging

### Check GPU Memory
```bash
# Linux/macOS
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

### Check CUDA Availability
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Test Model Loading
```bash
python -c "
from src.model import CustomAST
from torch import randn
model = CustomAST()
x = randn(1, 1, 128, 100)
output = model(x)
print(f'Model OK. Output shape: {output.shape}')
"
```

---

## 📊 Commonly Used Commands

### Check Data
```bash
# File size
du -sh data/icbhi_ast_16k_8s_metadata.npz

# Data info
python -c "
import numpy as np
data = np.load('data/icbhi_ast_16k_8s_metadata.npz')
print('Keys:', list(data.keys()))
print('X_train shape:', data['X_train'].shape)
print('y_train shape:', data['y_train'].shape)
print('Class distribution:', dict(zip(*np.unique(data['y_train'], return_counts=True))))
"
```

### Check Model
```bash
# Model summary
python -c "
from src.model import CustomAST
model = CustomAST()
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters: {total:,}')
print(f'Trainable: {trainable:,}')
"
```

### Clean Up
```bash
# Remove cached preprocessed data
rm data/icbhi_ast_16k_8s_metadata.npz

# Remove old checkpoints
rm -rf checkpoints/*

# Remove results
rm -rf results/*
```

---

## 🎯 Recommended Workflow

### For First-Time Users
```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess
python preprocess.py

# 3. Quick train (test)
python train.py --epochs 1 --batch_size 4

# 4. Full training
python train.py

# 5. Evaluate
python evaluate.py
```

### For Ablation Studies
```bash
# 1. Train multiple models
python train.py --use_sam --checkpoint_dir ./checkpoints/ast_sam
python train.py --no_sam --checkpoint_dir ./checkpoints/ast_adamw

# 2. Evaluate all
for dir in ./checkpoints/*/; do
    python evaluate.py \
        --model_path $dir/best_model.pth \
        --output_dir ./results/$(basename $dir)
done

# 3. Compare results
# Check results/*.png files
```

### For Hyperparameter Tuning
```bash
# Try different learning rates
for lr in 1e-6 5e-6 1e-5 5e-5; do
    python train.py \
        --lr $lr \
        --checkpoint_dir ./checkpoints/lr_${lr}
done

# Try different batch sizes
for bs in 4 8 16 32; do
    python train.py \
        --batch_size $bs \
        --checkpoint_dir ./checkpoints/bs_${bs}
done
```

---

## 💡 Performance Tips

### Speed Up Training
```bash
# Use FP16 (default with --use_amp)
python train.py --use_amp

# Larger batch size (if memory allows)
python train.py --batch_size 16

# Multiple GPU (if available)
# Note: Requires DataParallel wrapping (not implemented)
```

### Reduce Memory Usage
```bash
# Smaller batch
python train.py --batch_size 4

# Disable mixed-precision
python train.py --no_amp

# Shorter sequences in preprocessing
python preprocess.py --duration 6
```

---

## ✅ Final Checklist

- [ ] Environment activated
- [ ] Dependencies installed
- [ ] Data downloaded and extracted
- [ ] Preprocessing completed
- [ ] Training started successfully
- [ ] Best model saved
- [ ] Evaluation completed
- [ ] Visualizations generated
- [ ] Results examined

---

## 🆘 Quick Troubleshooting

```bash
# CUDA issues
python train.py --no_amp  # Force FP32

# Out of memory
python train.py --batch_size 4

# Slow training
# Check: nvidia-smi (GPU usage should be >90%)

# Model not improving
# Increase --epochs to 50
# Try different --lr (5e-6 or 5e-5)

# Data not found
# Run: python preprocess.py
# Check: ls -lh data/icbhi_ast_16k_8s_metadata.npz
```

---

**Ready to train! Start with: `python preprocess.py && python train.py && python evaluate.py`** 🚀
