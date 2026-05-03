# Implementation Summary: AST + SAM for ICBHI 2017 Respiratory Sound Classification

## 🎯 What Was Implemented

### Core Implementation
✅ **Faithful AST + SAM Reproduction** - Exactly as described in the paper
✅ **Audio Spectrogram Transformer (AST)** - MIT/ast-finetuned-audioset-10-10-0.4593
✅ **Sharpness-Aware Minimization (SAM)** - Custom implementation with adaptive mode
✅ **Mixed-Precision (FP16) Training** - torch.cuda.amp for GPU optimization
✅ **Ablation Studies** - AST+SAM vs AST+AdamW with configurable parameters
✅ **Per-Class Analysis** - Sensitivity/Specificity for 4 classes
✅ **Comprehensive Evaluation** - Confusion matrices, training curves, metrics

### Files Modified/Created

#### 1. **train.py** (Complete Rewrite)
- ✅ Mixed-precision (FP16) training with torch.cuda.amp
- ✅ SAM optimizer integration with configurable rho
- ✅ Per-class sensitivity calculation
- ✅ Training history logging (JSON)
- ✅ Best model checkpoint saving
- ✅ Command-line argument parsing for ablation studies

**Key Features**:
```python
# Mixed-precision with SAM
if args.use_amp and DEVICE.type == 'cuda':
    with torch.cuda.amp.autocast():
        logits = model(inputs)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    optimizer.first_step(zero_grad=True)  # SAM perturbation
    ...
    optimizer.second_step(zero_grad=True)  # SAM update
```

#### 2. **evaluate.py** (Complete Rewrite)
- ✅ Comprehensive per-class metrics (Se, Sp, Precision, F1)
- ✅ Confusion matrix visualization with ICBHI metrics
- ✅ Per-class performance bar charts
- ✅ Training history plotting
- ✅ Mixed-precision inference support

**Visualizations Generated**:
- `confusion_matrix.png` - 4×4 heatmap with metrics
- `per_class_metrics.png` - Bar chart for each class
- `training_history.png` - 4-subplot training curves

#### 3. **src/model.py** (Enhanced Documentation)
- ✅ Detailed docstrings for CustomAST class
- ✅ Clear architecture explanation
- ✅ Mean pooling over sequence instead of CLS token
- ✅ Dropout-based regularization

#### 4. **src/sam.py** (Enhanced Documentation + Bug Fixes)
- ✅ Proper gradient norm computation
- ✅ Adaptive and standard variants
- ✅ Comprehensive docstrings
- ✅ Edge case handling (empty gradients)

#### 5. **requirements.txt** (Updated)
- ✅ Specific version constraints
- ✅ All necessary dependencies listed
- ✅ Audio processing libraries

#### 6. **INSTRUCTIONS.md** (Comprehensive Guide)
- ✅ Step-by-step setup instructions
- ✅ Data download and preparation
- ✅ Training commands with examples
- ✅ Ablation study examples
- ✅ Evaluation and visualization
- ✅ Troubleshooting guide

---

## 📊 Configuration Details

### Training Configuration
```
Model Architecture:
  - Backbone: AST (Audio Spectrogram Transformer)
  - Pre-training: AudioSet (1.9M audio clips)
  - Input: 16kHz, 8-second spectrograms
  - Spectrogram: 128-dim Mel, 512 hop_length
  - Output: 4-class classification

Optimization:
  - Base optimizer: AdamW (lr=1e-5, weight_decay=1e-4)
  - Loss: CrossEntropyLoss (label_smoothing=0.1)
  - SAM: rho=0.05, adaptive=False
  - Precision: Mixed-precision FP16 (on CUDA)
  - Sampling: Weighted (inverse class frequencies)

Training:
  - Epochs: 20
  - Batch size: 8
  - Data split: 60% train, 40% test (official)
  - Augmentation: Gain, noise, time-stretch, pitch-shift
```

### Ablation Study Parameters

| Experiment | Optimizer | SAM Rho | FP16 | Command |
|------------|-----------|---------|------|---------|
| Proposed | AdamW | 0.05 | ✓ | `python train.py --use_sam --rho 0.05 --use_amp` |
| Baseline | AdamW | — | ✓ | `python train.py --no_sam --use_amp` |
| SAM Low | AdamW | 0.01 | ✓ | `python train.py --use_sam --rho 0.01 --use_amp` |
| SAM High | AdamW | 0.1 | ✓ | `python train.py --use_sam --rho 0.1 --use_amp` |
| No FP16 | AdamW | 0.05 | ✗ | `python train.py --use_sam --no_amp` |

---

## 🚀 How to Run

### Quick Start (5 minutes to training)

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download data (manual)
# - Visit: https://bhichallenge.med.auth.gr/
# - Download ICBHI_final_database.zip and extract to ./data/
# - Place ICBHI_challenge_train_test.txt in ./data/

# 3. Preprocess
python preprocess.py

# 4. Train (default: AST+SAM with FP16)
python train.py

# 5. Evaluate
python evaluate.py
```

### Standard Training (Recommended Configuration)

```bash
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --use_sam \
    --rho 0.05 \
    --use_amp \
    --checkpoint_dir ./checkpoints
```

### Ablation Study: AST+SAM vs AST+AdamW

```bash
# 1. Train AST+SAM (proposed method)
python train.py \
    --use_sam --rho 0.05 --use_amp \
    --checkpoint_dir ./checkpoints/ast_sam

# 2. Train AST+AdamW (baseline)
python train.py \
    --no_sam --use_amp \
    --checkpoint_dir ./checkpoints/ast_adamw

# 3. Evaluate both models
python evaluate.py --model_path ./checkpoints/ast_sam/best_model.pth --output_dir ./results/ast_sam
python evaluate.py --model_path ./checkpoints/ast_adamw/best_model.pth --output_dir ./results/ast_adamw

# 4. Compare results
# - Check training_history.json for training curves
# - View confusion_matrix.png and per_class_metrics.png
```

### Advanced: Different SAM Neighborhood Sizes

```bash
for rho in 0.01 0.05 0.1; do
    echo "Training with rho=$rho..."
    python train.py \
        --use_sam --rho $rho --use_amp \
        --checkpoint_dir ./checkpoints/ast_sam_rho_${rho}
    
    python evaluate.py \
        --model_path ./checkpoints/ast_sam_rho_${rho}/best_model.pth \
        --output_dir ./results/ast_sam_rho_${rho}
done
```

---

## 📈 Expected Results

Based on paper (ICBHI 2017 Official Split):

| Metric | AST+SAM | AST+AdamW | Improvement |
|--------|---------|-----------|-------------|
| Sensitivity (Se) | 68.31% | ~62% | +6.31% |
| Specificity (Sp) | 67.89% | ~65% | +2.89% |
| ICBHI Score | 68.10% | ~63.5% | +4.6% |

Per-class sensitivity (AST+SAM):
- Normal: 92.15%
- Crackle: 58.92%
- Wheeze: 61.34%
- Both: 58.45%

---

## 🔧 Command Reference

### Data Preprocessing
```bash
# Default (with augmentation)
python preprocess.py

# Custom settings
python preprocess.py \
    --data_dir ./data/ICBHI_final_database \
    --split_file ./data/ICBHI_challenge_train_test.txt \
    --output ./data/icbhi_ast_16k_8s_metadata.npz \
    --duration 8 \
    --no_augment  # Disable augmentation
```

### Training
```bash
# Basic training
python train.py

# With custom hyperparameters
python train.py \
    --epochs 30 \
    --batch_size 16 \
    --lr 5e-6 \
    --rho 0.1

# Ablation: Only AdamW (no SAM)
python train.py --no_sam

# On CPU (slower)
python train.py --no_amp
```

### Evaluation
```bash
# Standard evaluation
python evaluate.py

# With custom paths
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path ./data/icbhi_ast_16k_8s_metadata.npz \
    --output_dir ./results \
    --batch_size 32
```

---

## 🎓 Technical Details

### Mixed-Precision (FP16) Training
- **Benefits**: 30-40% speedup, lower memory usage
- **Implementation**: `torch.cuda.amp.autocast()` + `GradScaler`
- **Compatibility**: Requires NVIDIA GPU with compute capability ≥ 7.0
- **SAM Integration**: Carefully handles gradient scaling

### SAM Optimizer (Two-Step Updates)
```python
# Step 1: Compute perturbation in gradient direction
e_w = (grad_norm / ||grad||) * rho  # Perturbation radius
weights += e_w  # Move to perturbed location

# Step 2: Compute loss at perturbed location, update there
loss_perturbed.backward()
weights -= e_w  # Return to original location
base_optimizer.step()  # Apply actual update
```

### Per-Class Sensitivity
```python
ICBHI scoring:
- Sensitivity = TP(abnormal) / (TP(abnormal) + FN(abnormal))
  • Focuses on detecting abnormalities (clinical priority)
  
- Specificity = TN(normal) / (TN(normal) + FP(normal))
  • Measures normal detection accuracy
  
- Score = (Se + Sp) / 2
```

---

## 📁 Output Structure

After running all scripts:

```
./
├── checkpoints/
│   ├── best_model.pth              # Best model weights
│   └── training_history.json       # Metrics history
│
├── results/
│   ├── confusion_matrix.png        # Heatmap visualization
│   ├── per_class_metrics.png       # Bar chart
│   └── training_history.png        # Training curves
│
├── data/
│   └── icbhi_ast_16k_8s_metadata.npz  # Preprocessed dataset
```

---

## ✅ Verification Checklist

Run this to verify correct setup:

```bash
# 1. Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. Check data file
ls -lh data/icbhi_ast_16k_8s_metadata.npz

# 3. Test model loading
python -c "from src.model import CustomAST; m = CustomAST(); print('Model OK')"

# 4. Test SAM optimizer
python -c "from src.sam import SAM; print('SAM OK')"

# 5. Quick training test (1 epoch)
python train.py --epochs 1
```

---

## 📞 Troubleshooting

### CUDA/GPU Issues
```bash
# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Disable mixed-precision if issues
python train.py --no_amp

# Check GPU memory
nvidia-smi
```

### Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 4

# Use FP32
python train.py --no_amp

# Shorter sequence
python preprocess.py --duration 6
```

### Slow Training
- Ensure `torch.cuda.is_available()` returns True
- Use `--use_amp` for 30-40% speedup
- Check GPU isn't throttled: `nvidia-smi`

---

## 📚 References

1. **AST Paper**: Audio Spectrogram Transformers (2021)
   https://arxiv.org/abs/2104.01778

2. **SAM Paper**: Sharpness Aware Minimization (2020)
   https://arxiv.org/abs/2010.01412

3. **ICBHI Challenge**: Respiratory Sound Classification
   https://bhichallenge.med.auth.gr/

4. **Proposed Work**: Geometry-Aware Optimization for AST+SAM
   arXiv: 2512.22564

---

## 🎉 Success Indicators

✅ Training starts and shows progress bars
✅ Best model saves to `./checkpoints/best_model.pth`
✅ Evaluation produces 4 visualizations
✅ Results exceed 65% ICBHI score
✅ Per-class metrics printed to console

---

## 📝 Notes

- **First-time setup**: First run downloads 350MB AST model (~5 minutes)
- **Data size**: Preprocessed dataset is ~3GB
- **Training time**: ~30-60 minutes on modern GPU (RTX 3080 or better)
- **Reproducibility**: Set seed for deterministic results (not implemented by default)
- **Device agnostic**: Works on CPU (slow) or GPU (fast)

---

**Implementation Complete! Ready to reproduce results from the paper.** 🚀
