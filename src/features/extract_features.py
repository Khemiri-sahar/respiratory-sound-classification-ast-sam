import sys
import os
import argparse
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import gc
import glob
warnings.filterwarnings('ignore')
 
from src.features.scalogram_emd_cwt import EMDCWTExtractorFast
from src.features.mfcc_features import MFCCExtractor
 
 
def load_labels_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    y_train = data['y_train']
    y_test = data['y_test']
    all_labels = np.concatenate([y_train, y_test])
    
    print(f" Métadonnées chargées")
    print(f"   - Train: {len(y_train)}, Test: {len(y_test)}, Total: {len(all_labels)}")
    return all_labels
 
 
def find_wav_files(audio_dir):
    wav_files = glob.glob(os.path.join(audio_dir, "**/*.wav"), recursive=True)
    return sorted(wav_files)
 
 
def process_dataset(audio_dir, labels, output_dir, batch_size=50, use_emd=False):
"""
    Args:
        audio_dir (str): Directory containing .wav files
        labels (np.array): Labels array
        output_dir (str): Output directory for .npz files
        batch_size (int): Batch size for processing
        use_emd (bool): Use true EMD or filterbank
"""
    
    print(f"EXTRACTION - {'EMD PUR' if use_emd else 'FILTERBANK (RAPIDE)'}")
    
    # Find wav files
    print(f"\n Recherche .wav dans {audio_dir}...")
    wav_files = find_wav_files(audio_dir)
    
    if len(wav_files) == 0:
        print(f" Aucun .wav trouvé")
        return None, None, None
    
    print(f" {len(wav_files)} fichiers trouvés")
    
    # Match with labels
    n_samples = min(len(wav_files), len(labels))
    wav_files = wav_files[:n_samples]
    labels = labels[:n_samples]
    
    print(f" Traitement de {n_samples} échantillons")
    
    # Initialize extractors
    print(f"\n Initialisation des extracteurs (use_emd={use_emd})...")
    emd_cwt_extractor = EMDCWTExtractorFast(
        sr=16000, 
        n_imfs=5, 
        target_shape=(128, 128), 
        use_emd=use_emd
    )
    mfcc_extractor = MFCCExtractor(
        sr=16000, 
        n_mfcc=13, 
        target_shape=(128, 128)
    )
    
    # Process
    scalograms = []
    mfccs = []
    valid_labels = []
    errors = 0
    
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"\n Traitement en {num_batches} batchs...\n")
    
    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        
        print(f"Batch {batch_num + 1}/{num_batches}: {batch_start}-{batch_end}")
        
        for i in tqdm(range(batch_start, batch_end), desc=f"Batch {batch_num+1}"):
            try:
                # Load audio
                audio, _ = librosa.load(wav_files[i], sr=16000)
                
                # Extract features
                scalo = emd_cwt_extractor.extract(audio)
                mfcc = mfcc_extractor.extract(audio)
                
                scalograms.append(scalo)
                mfccs.append(mfcc)
                valid_labels.append(labels[i])
                
            except Exception as e:
                errors += 1
                print(f"\n Erreur échantillon {i}: {e}")
                continue
        
        # Checkpoint every 2 batches
        if (batch_num + 1) % 2 == 0:
            checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
            np.savez_compressed(
                checkpoint_path,
                scalograms=np.array(scalograms),
                mfccs=np.array(mfccs),
                labels=np.array(valid_labels)
            )
            print(f"💾 Checkpoint: {len(valid_labels)}/{n_samples}")
        
        gc.collect()
    
    # Convert to arrays
    scalograms = np.array(scalograms)
    mfccs = np.array(mfccs)
    valid_labels = np.array(valid_labels)
    
    print(f" EXTRACTION TERMINÉE")
    print(f"   Traités: {len(valid_labels)}/{n_samples}")
    print(f"   Erreurs: {errors}")
    print(f"   Scalograms: {scalograms.shape}")
    print(f"   MFCCs: {mfccs.shape}")
    
    # Save final files
    print("\n Sauvegarde des fichiers finaux...")
    
    scalo_path = os.path.join(output_dir, 'scalograms_filterbank_cwt_FINAL.npz')
    np.savez_compressed(scalo_path, scalograms=scalograms, labels=valid_labels)
    print(f" {scalo_path}")
    
    mfcc_path = os.path.join(output_dir, 'mfcc_features_FINAL.npz')
    np.savez_compressed(mfcc_path, features=mfccs, labels=valid_labels)
    print(f" {mfcc_path}")
    
    # Visualizations
    visualize_results(scalograms, mfccs, valid_labels, output_dir)
    
    # Clean checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\n  Checkpoint temporaire supprimé")
    
    return scalograms, mfccs, valid_labels
 
 
def visualize_results(scalograms, mfccs, labels, output_dir):
    
    print("\n Génération des visualisations...")
    
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    
    # 1. Comparison figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    for class_id in range(4):
        indices = np.where(labels == class_id)[0]
        if len(indices) >= 2:
            selected = np.random.choice(indices, 2, replace=False)
            
            for i, idx in enumerate(selected):
                # Scalogram
                ax = axes[class_id, i*2]
                im = ax.imshow(scalograms[idx], aspect='auto', origin='lower', cmap='jet')
                ax.set_title(f'{class_names[class_id]} - Scalogram', fontsize=10)
                plt.colorbar(im, ax=ax)
                
                # MFCC
                ax = axes[class_id, i*2+1]
                im = ax.imshow(mfccs[idx], aspect='auto', origin='lower', cmap='coolwarm')
                ax.set_title(f'{class_names[class_id]} - MFCCs', fontsize=10)
                plt.colorbar(im, ax=ax)
    
    plt.suptitle('Représentations Extraites - 2 échantillons par classe', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    comp_path = os.path.join(output_dir, 'representations_FINAL.png')
    plt.savefig(comp_path, dpi=150, bbox_inches='tight')
    print(f" {comp_path}")
    plt.close()
    
    # 2. Distribution
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar([class_names[l] for l in unique], counts, 
            color=['green', 'orange', 'red', 'purple'])
    plt.title('Distribution des Classes', fontsize=14, fontweight='bold')
    plt.ylabel('Nombre d\'échantillons')
    plt.xlabel('Classe')
    
    for i, (label, count) in enumerate(zip(unique, counts)):
        plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    dist_path = os.path.join(output_dir, 'distribution_FINAL.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f" {dist_path}")
    plt.close()
 
 
def main():    
    parser = argparse.ArgumentParser(
        description='Extract time-frequency features from ICBHI dataset'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='data/processed/icbhi_ast_16k_8s_metadata.npz',
        help='Path to metadata .npz file'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='data/raw/ICBHI_2017/audio_and_txt_files',
        help='Directory containing .wav files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for features'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--use-emd',
        action='store_true',
        help='Use true EMD instead of filterbank (very slow!)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print(" Configuration:")
    print(f"   Métadonnées: {args.metadata}")
    print(f"   Audio dir: {args.audio_dir}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Méthode: {'EMD pur (lent)' if args.use_emd else 'FILTERBANK (rapide)'}")
    
    if not args.use_emd:
        print("\n Utilisation du FILTERBANK")
    else:
        print("\n EMD pur activé")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load labels
    print()
    labels = load_labels_from_npz(args.metadata)
    
    # Print distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    print(f"\n Distribution:")
    for label, count in zip(unique, counts):
        print(f"   {class_names[label]:10s}: {count:4d} ({count/len(labels)*100:.1f}%)")
    
    # Extract features
    scalograms, mfccs, final_labels = process_dataset(
        audio_dir=args.audio_dir,
        labels=labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_emd=args.use_emd
    )
    
 
if __name__ == "__main__":
    main()