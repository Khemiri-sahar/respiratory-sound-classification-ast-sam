# Respiratory Sound Classification — ICBHI 2017

This repository contains multiple approaches and experiments for respiratory sound classification on the ICBHI 2017 dataset, including preprocessing pipelines, imbalance-aware training strategies, pretrained transformers, and CNN-based audio models.

---

# Table of Contents

1. Preprocess
2. Représentations Temps-Fréquence
3. Imbalance Losses
4. Pre-training
5. Modèle AST
6. CNN6

---

# 1. Preprocess

## ICBHI Preprocessing Pipeline

*Note: This project was built on top of another project.*

## preprocess.py — Pipeline complet

### Ce qui a été ajouté / modifié

| Fonctionnalité | Détail |
| :--- | :--- |
| **Spectrogrammes Mel** | `compute_mel()` → `librosa.feature.melspectrogram` + `power_to_db`, shape `(n_mels, T_frames)` |
| **Normalisation** | `normalize_mel()` → z-score (μ=0, σ=1) ou min-max [0,1], par sample |
| **Augmentation** | `augment_wav()` → bruit gaussien (0.5 % RMS) + time stretch ([0.85–1.15]) + pitch shift (±2 demi-tons), uniquement sur Crackle/Wheeze/Both en train |
| **Export .npz** | `X_train`/`X_test` (waveforms bruts pour AST) + `mel_train`/`mel_test` (spectros pour CNN/autres) + `class_weights` (poids inverse-fréquence) |
| **CLI argparse** | Tous les paramètres configurables sans toucher au code |

---

## Utilisation

```bash
# Défaut (128 Mel, hop=512, fmax=8kHz, z-score, augmentation activée)
python preprocess.py

# Paramètres personnalisés
python preprocess.py \
  --n_mels 64 \
  --hop_length 256 \
  --fmax 4000 \
  --norm_type minmax \
  --no_augment

# Choisir un autre répertoire de sortie
python preprocess.py --output ./data/icbhi_mel64.npz
```

---

## Format de sortie du `.npz`

```python
X_train      # (N_train, 128000)
mel_train    # (N_train, 128, 251)
y_train      # (N_train,)
device_train # (N_train,)
X_test / mel_test / y_test / device_test
class_weights # (4,)
```

---

# 2. Représentations Temps-Fréquence

> Cette section sera complétée ultérieurement.

---

# 3. Imbalance Losses

Compact guide for using the implemented imbalance methods.

## Files

```text
src/losses/weighted_ce.py
src/losses/focal_loss.py
src/losses/fbeta_loss.py
src/utils/threshold.py
train.py
evaluate.py
```

The loss files are not run directly. Use them through `train.py`.

---

## Required Input

Preprocessing must generate:

```text
data/icbhi_preprocessed.npz
```

---

## Quick Test

```powershell
python train.py --epochs 1 --limit_train_batches 2 --limit_eval_batches 2 --run_name debug_imbalance
python evaluate.py --model_path ./checkpoints/debug_imbalance.pth --limit_batches 2 --output_dir ./results/debug_imbalance
```

---

## Implemented Methods

### Weighted Cross-Entropy

Weighted Cross-Entropy gives more importance to rare classes during loss calculation.

### Focal Loss

Focal Loss focuses training on hard examples.

Recommended gamma values:

```text
0.5, 1.0, 2.0
```

### F-beta Loss

F-beta Loss emphasizes recall, especially for abnormal respiratory classes.

### Oversampling

Oversampling changes how training samples are selected.

### Threshold Tuning

Threshold tuning modifies class decision thresholds after training.

---

## Training Options

### Loss Options

```text
--loss ce
--loss weighted_ce
--loss focal
--loss fbeta
```

### Sampler Options

```text
--sampler weighted
--sampler minority
--sampler none
```

---

## Recommended Experiments

| Method | Command |
|---|---|
| Baseline | `python train.py --loss ce --sampler weighted --run_name baseline_ce_weighted` |
| Weighted CE | `python train.py --loss weighted_ce --sampler none --run_name weighted_ce` |
| Focal gamma 2.0 | `python train.py --loss focal --focal_gamma 2.0 --sampler none --run_name focal_g2` |
| F-beta beta 2 | `python train.py --loss fbeta --fbeta_beta 2 --fbeta_abnormal_only --sampler none --run_name fbeta2` |

---

# 4. Pre-training

## AudioSet Pretraining

The project relies on pretrained audio models initialized using large-scale AudioSet checkpoints.

Pretraining on AudioSet enables the models to learn robust acoustic representations before being fine-tuned on respiratory sound classification.

The pretrained weights are reused for:

- AST (Audio Spectrogram Transformer)
- CNN6 PANNs

Benefits:

- Better feature extraction
- Faster convergence
- Improved generalization
- Better performance on minority respiratory classes

---

# 5. Modèle AST

## AST + SAM for ICBHI 2017 Respiratory Sound Classification

### Core Implementation

✅ Audio Spectrogram Transformer (AST)
✅ Sharpness-Aware Minimization (SAM)
✅ Mixed-Precision FP16 Training
✅ Per-Class Sensitivity Analysis
✅ Confusion Matrix & Metrics

---

## Files

```text
train.py
evaluate.py
src/model.py
src/sam.py
```

---

## Training Configuration

```text
Backbone: AST
Pre-training: AudioSet
Input: 16kHz audio
Mel bins: 128
Epochs: 20
Batch size: 8
Optimizer: AdamW + SAM
```

---

## Training Example

```bash
python train.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --use_sam \
    --rho 0.05 \
    --use_amp
```

---

## Evaluation Metrics

```text
Sensitivity
Specificity
ICBHI Score
Recall per class
Confusion Matrix
```

ICBHI score:

\[
ICBHI\ Score = \frac{Sensitivity + Specificity}{2}
\]

---

## Outputs

```text
confusion_matrix.png
per_class_metrics.png
training_history.png
metrics_summary.csv
```

---

# 6. CNN6

## CNN6 PANNs Implementation

This repository also includes an implementation of CNN6 from the PANNs (Pretrained Audio Neural Networks) family.

CNN6 was adapted for respiratory sound classification using mel-spectrogram inputs generated during preprocessing.

---

## CNN6 Architecture

Input:

```text
(1, 128, 251)
```

Architecture:

```text
Mel-Spectrogram
        ↓
ConvBlock1 (64)
        ↓
ConvBlock2 (128)
        ↓
ConvBlock3 (256)
        ↓
ConvBlock4 (512)
        ↓
Global Pooling
        ↓
Fully Connected (512 → 512)
        ↓
Classifier (512 → 256 → 4)
        ↓
Final Classification
```

---

## Detailed Layer Configuration

| Layer | Input Shape | Output Shape | Details |
|---|---|---|---|
| ConvBlock1 | (1, 128, 251) | (64, 64, 125) | Conv2D(5×5) + BN + ReLU + AvgPool |
| ConvBlock2 | (64, 64, 125) | (128, 32, 62) | Conv2D(5×5) + BN + ReLU + AvgPool |
| ConvBlock3 | (128, 32, 62) | (256, 16, 31) | Conv2D(5×5) + BN + ReLU + AvgPool |
| ConvBlock4 | (256, 16, 31) | (512, 8, 15) | Conv2D(5×5) + BN + ReLU + AvgPool |
| Global Pooling | (512, 8, 15) | (512,) | Mean + Max pooling |
| FC1 | (512,) | (512,) | Linear + ReLU |
| Classifier | (512,) | (4,) | Linear + Dropout + Linear(4) |

---

## Pretrained Weights

Official checkpoint used:

```text
Cnn6_mAP=0.343.pth
```

Verification:

```text
Missing keys: []
Unexpected keys: 3
```

The unexpected keys correspond to the original spectrogram extractor layers, ignored because mel-spectrograms are already generated in preprocessing.

---

## Training Improvements

### Focal Loss

```text
gamma = 2
alpha = class_weights
```

### Progressive Fine-Tuning

| Module | Learning Rate |
|---|---|
| CNN6 Backbone | `1e-5` |
| Classification Head | `1e-4` |

### Scheduler

```text
CosineAnnealingLR
20 epochs
```

---

## CNN6 Results

| Epoch | Loss | Recall Macro | Sensitivity | Specificity | ICBHI Score |
|---|---|---|---|---|---|
| 1 | 0.7601 | 0.3075 | 0.4860 | 0.5332 | 0.5096 |
| 2 | 0.5928 | 0.3328 | 0.3356 | 0.6681 | 0.5019 |
| 5 | 0.4373 | 0.3481 | 0.3398 | 0.7061 | 0.5230 |
| 20 | 0.3095 | 0.3531 | 0.3806 | 0.6922 | 0.5364 |

Best checkpoint:

```text
Best Epoch: 19
Best ICBHI Score: 0.5426
```

---

## CNN6 Outputs

```text
cnn6_confusion_matrix.png
cnn6_training_history.png
cnn6_metrics_summary.csv
cnn6_predictions.csv
```

---

# References

1. AST Paper — Audio Spectrogram Transformers
2. SAM Paper — Sharpness Aware Minimization
3. ICBHI 2017 Respiratory Sound Classification Challenge
4. PANNs — Pretrained Audio Neural Networks

