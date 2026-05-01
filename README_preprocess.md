# ICBHI Preprocessing Pipeline

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

### Utilisation

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

### Format de sortie du `.npz`

```python
X_train      # (N_train, 128000)      
mel_train    # (N_train, 128, 251)    
y_train      # (N_train,)             
device_train # (N_train,)            
X_test / mel_test / y_test / device_test  # idem pour le test
class_weights # (4,)            
```