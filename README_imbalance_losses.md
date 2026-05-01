# Class Imbalance and Loss Functions

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

## Required Input

Preprocessing must generate:

```text
data/icbhi_preprocessed.npz
```

Default `train.py` and `evaluate.py` already use this path.

## Quick Test

Use this before long training:

```powershell
python train.py --epochs 1 --limit_train_batches 2 --limit_eval_batches 2 --run_name debug_imbalance
python evaluate.py --model_path ./checkpoints/debug_imbalance.pth --limit_batches 2 --output_dir ./results/debug_imbalance
```

This only checks that the code runs.

## Method Descriptions

### Weighted Cross-Entropy

Weighted Cross-Entropy gives more importance to rare classes during loss calculation. Each class receives a weight based on its frequency, so mistakes on rare classes such as Wheeze and Both are penalized more than mistakes on frequent classes. This method is useful when the model tends to favor the majority class.

Recommended first test:

```powershell
python train.py --loss weighted_ce --sampler none --run_name weighted_ce
```

### Focal Loss

Focal Loss focuses training on hard examples. Easy, confident predictions contribute less to the loss, while difficult or misclassified samples contribute more. The `gamma` value controls how strong this focusing effect is. Higher gamma values give more attention to hard samples.

Recommended gamma values:

```text
0.5, 1.0, 2.0
```

### F-beta Loss

F-beta Loss is used to emphasize recall. With `beta=2`, recall is more important than precision, which is useful when false negatives are costly. In this project, it can be applied mainly to abnormal classes using `--fbeta_abnormal_only`.

Recommended command:

```powershell
python train.py --loss fbeta --fbeta_beta 2 --fbeta_abnormal_only --sampler none --run_name fbeta2
```

### Oversampling

Oversampling changes how training samples are selected. Instead of changing the loss, it makes rare classes appear more often during training. The `weighted` sampler balances all classes using inverse frequency, while the `minority` sampler focuses especially on Wheeze and Both.

### Threshold Tuning

Threshold tuning is applied after training during evaluation. It changes the decision rule used to convert class probabilities into predicted labels. This can improve sensitivity by making the model less conservative when detecting abnormal classes. Threshold tuning should be reported carefully, especially if thresholds are selected using the test set.

### False-Negative Analysis

False-negative analysis identifies abnormal respiratory cycles that were predicted as Normal. These cases are important because they directly reduce sensitivity. The evaluation script saves these samples in `false_negatives_abnormal_as_normal.csv` for later inspection.

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

| Sampler | Meaning |
|---|---|
| `weighted` | inverse-frequency sampling for all classes |
| `minority` | extra focus on Wheeze and Both |
| `none` | no oversampling |

## Recommended Experiments

Run one experiment at a time.

| Method | Command |
|---|---|
| Baseline | `python train.py --loss ce --sampler weighted --run_name baseline_ce_weighted` |
| CE only | `python train.py --loss ce --sampler none --run_name ce_no_sampler` |
| Weighted CE | `python train.py --loss weighted_ce --sampler none --run_name weighted_ce` |
| Focal gamma 0.5 | `python train.py --loss focal --focal_gamma 0.5 --sampler none --run_name focal_g05` |
| Focal gamma 1.0 | `python train.py --loss focal --focal_gamma 1.0 --sampler none --run_name focal_g1` |
| Focal gamma 2.0 | `python train.py --loss focal --focal_gamma 2.0 --sampler none --run_name focal_g2` |
| F-beta beta 2 | `python train.py --loss fbeta --fbeta_beta 2 --fbeta_abnormal_only --sampler none --run_name fbeta2` |
| Minority oversampling | `python train.py --loss ce --sampler minority --run_name ce_minority_sampler` |

## Evaluation

Evaluate a checkpoint:

```powershell
python evaluate.py --model_path ./checkpoints/EXPERIMENT_NAME.pth --output_dir ./results/EXPERIMENT_NAME
```

Example:

```powershell
python evaluate.py --model_path ./checkpoints/weighted_ce.pth --output_dir ./results/weighted_ce
```

## Threshold Tuning

Tune thresholds after training:

```powershell
python evaluate.py --model_path ./checkpoints/focal_g2.pth --output_dir ./results/focal_g2_thresholds --tune_thresholds --threshold_objective icbhi --save_thresholds_path ./results/focal_g2_thresholds/thresholds.json
```

Other objectives:

```text
--threshold_objective icbhi
--threshold_objective sensitivity
--threshold_objective fbeta
```

Use manual thresholds:

```powershell
python evaluate.py --model_path ./checkpoints/focal_g2.pth --thresholds 0.5,0.4,0.4,0.3 --output_dir ./results/focal_g2_manual_thresholds
```

Use saved thresholds:

```powershell
python evaluate.py --model_path ./checkpoints/focal_g2.pth --thresholds_path ./results/focal_g2_thresholds/thresholds.json --output_dir ./results/focal_g2_saved_thresholds
```

## Evaluation Outputs

Each evaluation creates:

```text
confusion_matrix.png
metrics_summary.csv
predictions.csv
false_negatives_abnormal_as_normal.csv
normal_as_abnormal.csv
```

Important files:

| File | Use |
|---|---|
| `metrics_summary.csv` | final scores and class recalls |
| `confusion_matrix.png` | visual class performance |
| `false_negatives_abnormal_as_normal.csv` | abnormal cycles predicted as Normal |
| `normal_as_abnormal.csv` | Normal cycles predicted as abnormal |

## Colab / Kaggle

Use the same commands after uploading the repo and `data/icbhi_preprocessed.npz`.

Example:

```bash
python train.py --data_path ./data/icbhi_preprocessed.npz --loss focal --focal_gamma 2 --sampler none --epochs 20 --batch_size 8 --run_name focal_g2
python evaluate.py --data_path ./data/icbhi_preprocessed.npz --model_path ./checkpoints/focal_g2.pth --output_dir ./results/focal_g2
```

If GPU memory is low:

```bash
python train.py --loss focal --focal_gamma 2 --batch_size 4 --sampler none --run_name focal_g2_bs4
```

## Metrics To Report

```text
Sensitivity
Specificity
ICBHI Score
Recall per class
Abnormal -> Normal false negatives
Normal -> abnormal errors
```

## Fair Comparison

Keep fixed:

```text
same .npz file
same train/test split
same model
same epochs
same batch size
same learning rate
same SAM rho
```

Change only one imbalance method per experiment.
