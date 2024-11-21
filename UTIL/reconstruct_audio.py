# reconstruct_audio.py

import librosa
import numpy as np
import soundfile as sf
import os

def reconstruct_audio(mfccs, sr, output_path):
    # Convertir MFCCs a audio
    y_reconstructed = librosa.feature.inverse.mfcc_to_audio(mfccs, sr=sr)
    # Guardar el audio reconstruido
    sf.write(output_path, y_reconstructed, sr)
    print(f"Audio procesado guardado en: {output_path}")

# Esta función es una entrada para llamar a la reconstrucción desde otro script
def reconstruct(mfccs, sr, file_name):
    output_dir = "TEST/reconstruction"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, file_name)
    reconstruct_audio(mfccs, sr, output_path)
