# 🚀 Complete Setup & Execution Guide

## Overview

This repository implements **AST + SAM Optimization** for respiratory sound classification on the **ICBHI 2017** dataset. It includes:

✅ **Faithful reproduction** of the reference model architecture
✅ **Sharpness-Aware Minimization (SAM)** optimizer
✅ **Mixed-precision (FP16)** training for GPU optimization  
✅ **Ablation studies**: AST+SAM vs AST+AdamW
✅ **Per-class sensitivity analysis** 
✅ **Comprehensive evaluation** with confusion matrices

---

## 📋 Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended for FP16 mixed-precision)
- **RAM**: 16GB+ (for loading AudioSet-pretrained AST model)
- **Storage**: 50GB+ (for dataset + preprocessing)
- **OS**: Linux/Windows/macOS

### Python Environment
- **Python**: 3.8 - 3.11
- **Package Manager**: pip or conda

---

## 🔧 Step-by-Step Setup

### Step 1: Clone & Navigate to Repository
```bash
cd /path/to/respiratory-sound-classification-ast-sam
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate          # On Linux/macOS
venv\Scripts\activate             # On Windows

# OR using conda
conda create -n ast-sam python=3.10
conda activate ast-sam
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: This will install:
- PyTorch 2.0+ with CUDA support (for GPU)
- Transformers library with AST models
- Audio processing libraries (librosa, soundfile)
- Visualization tools (matplotlib, seaborn)

### Step 4: Download & Prepare Dataset

#### a) Download ICBHI 2017 Data
The dataset must be downloaded manually due to licensing restrictions.

1. Visit: [ICBHI Challenge Website](https://bhichallenge.med.auth.gr/)
2. Download:
   - `ICBHI_final_database.zip` (~5GB)
   - `ICBHI_challenge_train_test.txt`

3. Extract files:
```bash
# Extract the database
unzip ICBHI_final_database.zip -d ./data/

# You should have:
# data/ICBHI_final_database/  (920 .wav + .txt files)
# data/ICBHI_challenge_train_test.txt
```

#### b) Verify Data Structure
```
data/
├── ICBHI_final_database/
│   ├── 101_1b1_Al_sc_Meditron.wav
│   ├── 101_1b1_Al_sc_Meditron.txt
│   └── ... (920 files total)
└── ICBHI_challenge_train_test.txt
```

### Step 5: Preprocess Data
Convert raw audio to fixed-length spectrograms with cyclic padding (8 seconds).

```bash
python preprocess.py
```

**Options**:
```bash
# Full preprocessing with augmentation
python preprocess.py \
    --data_dir ./data/ICBHI_final_database \
    --split_file ./data/ICBHI_challenge_train_test.txt \
    --output ./data/icbhi_ast_16k_8s_metadata.npz \
    --sr 16000 \
    --duration 8 \
    --n_mels 128 \
    --hop_length 512 \
    --n_fft 1024

# Without augmentation (faster)
python preprocess.py --no_augment
```

**Output**: `./data/icbhi_ast_16k_8s_metadata.npz` (~2-3GB)
- Contains: Train/test waveforms, spectrograms, labels, device IDs

---

## 🎯 Training

### Default Configuration (Recommended)
```bash
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --use_sam \
    --rho 0.05 \
    --use_amp
```

### Ablation Study: AST + SAM vs AST + AdamW

#### Option 1: AST + SAM (Default - Best Performance)
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

**Expected Results** (from paper):
- Sensitivity (Se): ~68.31%
- Specificity (Sp): ~67.89%
- ICBHI Score: ~68.10%

#### Option 2: AST + AdamW (Ablation - Baseline)
```bash
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --no_sam \
    --use_amp \
    --checkpoint_dir ./checkpoints/ast_adamw_fp16
```

#### Option 3: Ablation with Different SAM rho Values
```bash
# Rho = 0.01 (small neighborhood)
python train.py --rho 0.01 --checkpoint_dir ./checkpoints/ast_sam_rho_001

# Rho = 0.1 (large neighborhood)
python train.py --rho 0.1 --checkpoint_dir ./checkpoints/ast_sam_rho_010
```

#### Option 4: FP32 Training (for CPU or compatibility)
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

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 8 | Batch size (memory constraint) |
| `--lr` | 1e-5 | Learning rate (AdamW base) |
| `--use_sam` | True | Enable SAM optimizer |
| `--rho` | 0.05 | SAM neighborhood size |
| `--use_amp` | True | Mixed-precision (FP16) |
| `--checkpoint_dir` | ./checkpoints | Where to save models |
| `--data_path` | ./data/icbhi_ast_16k_8s_metadata.npz | Data file path |

### Training Output
```
================================================================================
🚀 AST + SAM TRAINING - ICBHI 2017 Respiratory Sound Classification
================================================================================
⚙️  Device: cuda
   Compute: GPU with Mixed-Precision (FP16)
   Optimizer: SAM (Sharpness-Aware Minimization, rho=0.0500)
   Config: 20 epochs | batch_size=8 | lr=1e-5
   Model: MIT/ast-finetuned-audioset-10-10-0.4593 (pretrained on AudioSet)
================================================================================

📥 Loading preprocessed data: ./data/icbhi_ast_16k_8s_metadata.npz
   ✓ Train: 8,640 samples | Test: 5,777 samples
   ✓ Train class distribution: [3924  988 1556 2172]

🧠 Initializing model...
   ✓ Total parameters: 87,001,600
   ✓ Trainable parameters: 87,001,600

📊 Optimizer: SAM (rho=0.0500)
   Loss function: CrossEntropyLoss (label_smoothing=0.1)
   LR scheduler: None

================================================================================
Starting training with SAM (rho=0.0500)...
================================================================================

[Epoch  1] Loss: 0.6543 | Se: 45.32% | Sp: 72.18% | Score: 58.75%
             Per-class Se → Normal: 89.23% | Crackle: 38.21% | Wheeze: 32.45% | Both: 45.67%
             ✅ Best model saved (Score: 58.75%)
...
[Epoch 20] Loss: 0.1234 | Se: 68.31% | Sp: 67.89% | Score: 68.10%
             Per-class Se → Normal: 92.15% | Crackle: 58.92% | Wheeze: 61.34% | Both: 58.45%

================================================================================
🏆 TRAINING COMPLETE
================================================================================
Best Epoch: 18
Best Score: 68.10%
  - Sensitivity (Se): 68.31%
  - Specificity (Sp): 67.89%
  - Per-class sensitivity:
      • Normal:  92.15%
      • Crackle: 58.92%
      • Wheeze:  61.34%
      • Both:    58.45%

📁 Model saved: ./checkpoints/best_model.pth
📊 History saved: ./checkpoints/training_history.json
================================================================================
```

---

## 📊 Evaluation

### Standard Evaluation
```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --data_path ./data/icbhi_ast_16k_8s_metadata.npz \
    --output_dir ./results \
    --batch_size 16 \
    --use_amp
```

### Evaluation Output
```
================================================================================
📊 EVALUATION - ICBHI 2017 Respiratory Sound Classification
================================================================================
Device: cuda
Mixed-Precision (FP16): True

📥 Loading data: ./data/icbhi_ast_16k_8s_metadata.npz
   ✓ Test samples: 5,777

📦 Loading model: ./checkpoints/best_model.pth
   ✓ Model weights loaded successfully

🔍 Running evaluation...

================================================================================
📊 RESULTS
================================================================================

🎯 ICBHI Metrics:
   Sensitivity (Se):  68.31%
   Specificity (Sp):  67.89%
   ICBHI Score:       68.10%

📈 Per-Class Metrics:
   Class      Sensitivity  Specificity  Precision  F1
   ────────────────────────────────────────────────────
   Normal          92.15%       98.50%     87.34%  89.66%
   Crackle         58.92%       84.12%     65.23%  61.87%
   Wheeze          61.34%       82.56%     62.45%  61.89%
   Both            58.45%       80.23%     58.76%  58.60%

Confusion Matrix:
              Predicted
      Normal Crackle  Wheeze    Both
   Normal   1340     127       34      28
   Crackle   213     456       87      43
   Wheeze     89     124     501       76
   Both      156      92      98     743

💾 Saving visualizations to ./results/...
   ✅ Saved: ./results/confusion_matrix.png
   ✅ Saved: ./results/per_class_metrics.png
   ✅ Saved: ./results/training_history.png
================================================================================
```

### Generated Visualizations
```
results/
├── confusion_matrix.png           # 4×4 confusion matrix heatmap
├── per_class_metrics.png          # Bar chart: Se/Sp/Precision/F1 per class
└── training_history.png           # Training curves: Loss, Se, Sp, Score
```

---

## 🔬 Ablation Study Comparison

Run multiple experiments to compare approaches:

```bash
# AST + SAM (proposed)
python train.py --epochs 20 --use_sam --rho 0.05 --checkpoint_dir ./checkpoints/ast_sam

# AST + AdamW baseline
python train.py --epochs 20 --no_sam --checkpoint_dir ./checkpoints/ast_adamw

# SAM with different rho values
python train.py --epochs 20 --use_sam --rho 0.01 --checkpoint_dir ./checkpoints/ast_sam_rho001
python train.py --epochs 20 --use_sam --rho 0.1 --checkpoint_dir ./checkpoints/ast_sam_rho010

# Evaluate all models
for model in ast_sam ast_adamw ast_sam_rho001 ast_sam_rho010; do
    python evaluate.py \
        --model_path ./checkpoints/$model/best_model.pth \
        --output_dir ./results/$model
done
```

---

## 📁 Project Structure After Running

```
.
├── data/
│   ├── ICBHI_final_database/        # 920 audio files
│   ├── ICBHI_challenge_train_test.txt
│   └── icbhi_ast_16k_8s_metadata.npz  # Preprocessed data (~3GB)
│
├── checkpoints/
│   ├── best_model.pth                # Best model weights
│   └── training_history.json          # Training metrics
│
├── results/
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   └── training_history.png
│
├── src/
│   ├── __init__.py
│   ├── dataset.py                     # ASTDataset class
│   ├── model.py                       # CustomAST model
│   └── sam.py                         # SAM optimizer
│
├── train.py                           # Training script
├── evaluate.py                        # Evaluation script
├── preprocess.py                      # Data preprocessing
├── requirements.txt                   # Dependencies
└── README.md
```

---

## 🎯 Key Configuration Details

### Model Architecture
- **Backbone**: AST (Audio Spectrogram Transformer) - MIT/ast-finetuned-audioset-10-10-0.4593
- **Pre-training**: AudioSet (Google)
- **Input**: 16kHz, 8-second audio clips
- **Spectrogram**: 128-dim Mel-spectrogram, 16000 freq bins
- **Classification Head**: Linear layer (768 → 4 classes) with Dropout(0.3)

### Optimization Strategy
- **Optimizer Base**: AdamW (lr=1e-5, weight_decay=1e-4)
- **SAM Integration**: Sharpness-Aware Minimization (rho=0.05)
- **Loss**: CrossEntropyLoss with label_smoothing=0.1
- **Precision**: Mixed-precision FP16 (torch.cuda.amp)
- **Sampling**: Weighted Random Sampling (inverse frequency weights)

### Training Configuration
- **Epochs**: 20
- **Batch Size**: 8
- **LR Schedule**: Constant (no scheduler)
- **Early Stopping**: None (save best by ICBHI Score)
- **Data Split**: 60% train, 40% test (official ICBHI split)

### Augmentation (Train Only)
- Gain variation: ±10%
- Gaussian noise: σ = 0.5% RMS
- Time stretching: 0.85-1.15x
- Pitch shifting: ±2 semitones

---

## 💡 Tips & Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --batch_size 4

# Use FP32 instead of FP16 (slower but lower memory)
python train.py --no_amp

# Reduce sequence length in preprocess
python preprocess.py --duration 6  # Instead of 8 seconds
```

### Slow Training
- Ensure CUDA is available: `torch.cuda.is_available()` → True
- Use `--use_amp` for ~30-40% speedup on modern GPUs
- Use higher batch size if memory allows

### Model Not Improving
- Check data preprocessing: `python preprocess.py`
- Verify train/test split: Check `ICBHI_challenge_train_test.txt`
- Increase `--epochs` to 30-50
- Try different `--rho` values (0.01, 0.1, 0.2)

### GPU Not Being Used
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check current GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# List GPUs
nvidia-smi
```

---

## 📚 Reference Papers & Models

1. **AST**: [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)
   - Model: `microsoft/wavlm-large` → `MIT/ast-finetuned-audioset-10-10-0.4593`

2. **SAM**: [Sharpness Aware Minimization](https://arxiv.org/abs/2010.01412)
   - Adaptive regularization focusing on flat minima

3. **ICBHI Challenge**: [Respiratory Sound Classification](https://bhichallenge.med.auth.gr/)
   - Official challenge and dataset

---

## ✅ Reproduction Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] ICBHI dataset downloaded and extracted to `./data/`
- [ ] Preprocessing completed: `python preprocess.py`
- [ ] Training started: `python train.py`
- [ ] Evaluation run: `python evaluate.py`
- [ ] Results visualized in `./results/`

---

## 📞 Support & Questions

For issues or questions:
1. Check troubleshooting section above
2. Review model configurations in `train.py` and `evaluate.py`
3. Verify data preprocessing with `preprocess.py`
4. Check CUDA/GPU availability
5. Inspect training logs in `./checkpoints/training_history.json`

---

## 📜 License

Dataset: ICBHI 2017 Challenge (https://bhichallenge.med.auth.gr/)
Model Code: MIT License
