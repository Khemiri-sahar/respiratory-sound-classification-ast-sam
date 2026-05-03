"""
  1. Load .wav / .txt annotation pairs from ICBHI_final_database
  2. Segment respiratory cycles from annotated timestamps
  3. Cyclic padding to TARGET_DURATION (8 s) without information loss
  4. Compute log-Mel spectrograms (librosa) with configurable parameters
  5. Per-sample normalisation: z-score or min-max
  6. Data augmentation for rare classes (train split only):
       - Additive Gaussian noise
       - Time stretching  (rate ∈ [0.85, 1.15])
       - Pitch shifting   (n_steps ∈ [-2, +2] semitones)
  7. Export to .npz:
       X_train / X_test      — raw waveforms   (N, 128 000)
       mel_train / mel_test  — log-Mel spectra  (N, n_mels, T_frames)
       y_train / y_test      — integer labels   (N,)
       device_train / device_test — device IDs  (N,)
       class_weights          — inverse-frequency weights (4,)

Label encoding:  0=Normal  1=Crackle  2=Wheeze  3=Both
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

DATA_DIR        = "./data/ICBHI_final_database"
SPLIT_FILE      = "./data/ICBHI_Challenge_train_test.txt"
OUTPUT_NPZ      = "./data/icbhi_ast_16k_8s_metadata.npz"

TARGET_SR       = 16_000
TARGET_DURATION = 8                             
TARGET_SAMPLES  = TARGET_SR * TARGET_DURATION   # 128 000 samples

N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 1024
FMAX       = 8000
NORM_TYPE  = "zscore"   
DO_AUGMENT = True
RARE_LABELS = {1, 2, 3}  

DEVICE_MAP = {
    "AKGC417L": 0,
    "LittC2SE": 1,
    "Litt3200": 2,
    "Meditron": 3,
}
LABEL_NAMES = {0: "Normal", 1: "Crackle", 2: "Wheeze", 3: "Both"}


# ── Utility functions ─────────────────────────────────────────────────────────

def get_device_id(filename: str) -> int:
    return DEVICE_MAP.get(filename.split("_")[-1], -1)


def cyclic_padding(wav: np.ndarray, target_len: int) -> np.ndarray:
    """Tile the signal until target_len, then truncate — preserves temporal structure."""
    if len(wav) >= target_len:
        return wav[:target_len]
    return np.tile(wav, (target_len // len(wav)) + 1)[:target_len]


def compute_mel(wav: np.ndarray, cfg: argparse.Namespace) -> np.ndarray:
    """Return log-Mel spectrogram of shape (n_mels, T_frames) as float32."""
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmax=cfg.fmax,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def normalize_mel(mel: np.ndarray, norm_type: str) -> np.ndarray:
    """Per-sample normalisation of a (n_mels, T) spectrogram."""
    if norm_type == "zscore":
        return ((mel - mel.mean()) / (mel.std() + 1e-8)).astype(np.float32)
    if norm_type == "minmax":
        lo, hi = mel.min(), mel.max()
        return ((mel - lo) / (hi - lo + 1e-8)).astype(np.float32)
    return mel.astype(np.float32)


def augment_wav(wav: np.ndarray, cfg: argparse.Namespace) -> list:
    """
    Generate three augmented variants of a padded waveform:
      1. Gaussian noise  — scaled to 0.5 % of signal RMS
      2. Time stretching — rate ∈ [0.85, 1.15]  (re-padded if needed)
      3. Pitch shifting  — n_steps ∈ [-2, +2] semitones
    """
    out = []

    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-8
    noisy = wav + np.random.normal(0, 0.005 * rms, wav.shape).astype(np.float32)
    out.append(np.clip(noisy, -1.0, 1.0))

    rate = float(np.random.uniform(0.85, 1.15))
    stretched = librosa.effects.time_stretch(wav, rate=rate)
    out.append(cyclic_padding(stretched.astype(np.float32), cfg.target_samples))

    n_steps = float(np.random.uniform(-2.0, 2.0))
    shifted = librosa.effects.pitch_shift(wav, sr=cfg.sr, n_steps=n_steps)
    out.append(cyclic_padding(shifted.astype(np.float32), cfg.target_samples))

    return out


def build_label(crackle: int, wheeze: int) -> int:
    if crackle == 0 and wheeze == 0:
        return 0
    if crackle == 1 and wheeze == 0:
        return 1
    if crackle == 0 and wheeze == 1:
        return 2
    return 3


# ── Core pipeline ─────────────────────────────────────────────────────────────

def _append_sample(wav, mel, label, dev_id, X_list, mel_list, y_list, dev_list):
    X_list.append(wav)
    mel_list.append(mel)
    y_list.append(label)
    dev_list.append(dev_id)


def process_recording(
    fname: str,
    audio: np.ndarray,
    anns_df: pd.DataFrame,
    cfg: argparse.Namespace,
    is_train: bool,
    X_tr, mel_tr, y_tr, dev_tr,
    X_te, mel_te, y_te, dev_te,
):
    dev_id = get_device_id(fname)

    for _, ann in anns_df.iterrows():
        start = int(float(ann["start"]) * cfg.sr)
        end   = int(float(ann["end"])   * cfg.sr)
        chunk = audio[start:end]

        if len(chunk) < 100:
            continue

        wav   = cyclic_padding(chunk.astype(np.float32), cfg.target_samples)
        label = build_label(int(ann["crackle"]), int(ann["wheeze"]))
        mel   = normalize_mel(compute_mel(wav, cfg), cfg.norm_type)

        if is_train:
            _append_sample(wav, mel, label, dev_id, X_tr, mel_tr, y_tr, dev_tr)

            if cfg.augment and label in RARE_LABELS:
                for aug_wav in augment_wav(wav, cfg):
                    aug_mel = normalize_mel(compute_mel(aug_wav, cfg), cfg.norm_type)
                    _append_sample(aug_wav, aug_mel, label, dev_id, X_tr, mel_tr, y_tr, dev_tr)
        else:
            _append_sample(wav, mel, label, dev_id, X_te, mel_te, y_te, dev_te)


def process_data(cfg: argparse.Namespace):
    print("── ICBHI 2017 Preprocessing Pipeline ──────────────────────────────")
    print(f"  data_dir   : {cfg.data_dir}")
    print(f"  SR         : {cfg.sr} Hz  |  duration: {cfg.duration}s  ({cfg.target_samples} samples)")
    print(f"  Mel        : n_mels={cfg.n_mels}  hop_length={cfg.hop_length}  n_fft={cfg.n_fft}  fmax={cfg.fmax}")
    print(f"  Norm       : {cfg.norm_type}")
    print(f"  Augment    : {cfg.augment}  (rare labels: {sorted(RARE_LABELS)} → {[LABEL_NAMES[l] for l in sorted(RARE_LABELS)]})")
    print("────────────────────────────────────────────────────────────────────")

    if not os.path.exists(cfg.split_file):
        raise FileNotFoundError(f"Split file not found: {cfg.split_file}")
    if not os.path.isdir(cfg.data_dir):
        raise FileNotFoundError(f"Data directory not found: {cfg.data_dir}")

    split_df = pd.read_csv(cfg.split_file, sep="\t", names=["filename", "set_type"])

    X_tr, mel_tr, y_tr, dev_tr = [], [], [], []
    X_te, mel_te, y_te, dev_te = [], [], [], []
    skipped = 0

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Recordings"):
        fname    = str(row["filename"]).strip()
        set_type = str(row["set_type"]).strip()

        wav_path = os.path.join(cfg.data_dir, fname + ".wav")
        txt_path = os.path.join(cfg.data_dir, fname + ".txt")

        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            skipped += 1
            continue

        audio, _ = librosa.load(wav_path, sr=cfg.sr, mono=True)
        anns = pd.read_csv(txt_path, sep="\t", names=["start", "end", "crackle", "wheeze"])

        process_recording(
            fname, audio, anns, cfg, is_train=(set_type == "train"),
            X_tr=X_tr, mel_tr=mel_tr, y_tr=y_tr, dev_tr=dev_tr,
            X_te=X_te, mel_te=mel_te, y_te=y_te, dev_te=dev_te,
        )

    if skipped:
        print(f"\n  [!] {skipped} recording(s) skipped (file not found)")

    # ── Assemble numpy arrays ─────────────────────────────────────────────────
    X_train      = np.array(X_tr,   dtype=np.float32)
    mel_train    = np.array(mel_tr, dtype=np.float32)
    y_train      = np.array(y_tr,   dtype=np.int64)
    device_train = np.array(dev_tr, dtype=np.int64)

    X_test       = np.array(X_te,   dtype=np.float32)
    mel_test     = np.array(mel_te, dtype=np.float32)
    y_test       = np.array(y_te,   dtype=np.int64)
    device_test  = np.array(dev_te, dtype=np.int64)

    # Inverse-frequency class weights over the (augmented) training set
    counts = np.bincount(y_train, minlength=4).astype(np.float64)
    class_weights = (counts.sum() / (4.0 * counts + 1e-8)).astype(np.float32)

    # ── Statistics ────────────────────────────────────────────────────────────
    print("\n── Dataset statistics ──────────────────────────────────────────────")
    print(f"  Train : waveforms {X_train.shape}  |  mel {mel_train.shape}")
    print(f"  Test  : waveforms {X_test.shape}   |  mel {mel_test.shape}")
    print("\n  Train label distribution (after augmentation):")
    for lbl, name in LABEL_NAMES.items():
        n = int((y_train == lbl).sum())
        pct = 100.0 * n / max(len(y_train), 1)
        bar = "█" * (n * 30 // max(int(counts.max()), 1))
        print(f"    [{lbl}] {name:8s}: {n:5d} ({pct:5.1f}%)  {bar}")
    print(f"\n  Class weights : {class_weights}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = cfg.output
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Use uncompressed save to avoid corruption during write
    np.savez(
        out_path,
        X_train=X_train,   mel_train=mel_train,   y_train=y_train,   device_train=device_train,
        X_test=X_test,     mel_test=mel_test,      y_test=y_test,     device_test=device_test,
        class_weights=class_weights,
    )

    final_path = out_path if os.path.exists(out_path) else out_path + ".npz"
    size_mb = os.path.getsize(final_path) / 1e6
    print(f"\nSaved → {final_path}  ({size_mb:.1f} MB)")
    print("────────────────────────────────────────────────────────────────────")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICBHI 2017 respiratory audio preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",   default=DATA_DIR,        help="Path to ICBHI_final_database/")
    p.add_argument("--split_file", default=SPLIT_FILE,      help="Path to train/test split .txt")
    p.add_argument("--output",     default=OUTPUT_NPZ,      help="Output .npz path")

    p.add_argument("--sr",         type=int, default=TARGET_SR,       help="Target sample rate (Hz)")
    p.add_argument("--duration",   type=int, default=TARGET_DURATION, help="Padded duration (seconds)")

    p.add_argument("--n_mels",     type=int, default=N_MELS,     help="Number of Mel filter banks")
    p.add_argument("--hop_length", type=int, default=HOP_LENGTH, help="STFT hop length")
    p.add_argument("--n_fft",      type=int, default=N_FFT,      help="FFT window size")
    p.add_argument("--fmax",       type=int, default=FMAX,        help="Highest frequency for Mel (Hz)")

    p.add_argument("--norm_type",  default=NORM_TYPE, choices=["zscore", "minmax"],
                   help="Per-sample normalisation method")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable data augmentation for rare classes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.target_samples = args.sr * args.duration
    args.augment = not args.no_augment
    process_data(args)
