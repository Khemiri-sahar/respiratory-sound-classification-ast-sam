import librosa
import numpy as np
import cv2

class MFCCExtractor:
    """
    Extracteur de caractéristiques MFCC (Mel-frequency cepstral coefficients).
    Transforme un signal audio 1D en une représentation 2D (image).
    """
    def __init__(self, sr=16000, n_mfcc=13, target_shape=(128, 128)):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.target_shape = target_shape

    def extract(self, audio):
        """
        Extrait les MFCCs et redimensionne la matrice à la taille cible.
        """
        # 1. Extraction des MFCC de base
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        
        # 2. Redimensionnement (Interpolation) pour obtenir du (128, 128)
        # OpenCV est extrêmement rapide pour ça
        mfccs_resized = cv2.resize(
            mfccs, 
            (self.target_shape[1], self.target_shape[0]), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # 3. Conversion en float32
        return mfccs_resized.astype(np.float32)