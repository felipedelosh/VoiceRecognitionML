"""
FelipedelosH
"""
import librosa
import numpy as np
import soundfile as sf
import os

def reconstruct_audio(mfccs, sr, output_path):
    if np.any(np.isinf(mfccs)) or np.any(np.isnan(mfccs)): # No infinity or NaN in audio
        mfccs = np.nan_to_num(mfccs, nan=0.0, posinf=1.0, neginf=-1.0)  # Reeplace NaN Or Infinity
        mfccs = np.clip(mfccs, -1.0, 1.0)  # Limit values
    
    mfccs = mfccs.T
    y_reconstructed = librosa.feature.inverse.mfcc_to_audio(mfccs, sr=sr)
    
    sf.write(output_path, y_reconstructed, sr)
    print(f"Audio procesado guardado en: {output_path}")


def reconstruct(mfccs, sr, file_name):
    try:
        output_dir = "TEST/reconstruction"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, file_name)
        reconstruct_audio(mfccs, sr, output_path)
    except Exception as e:
        print(f"Error al procesar {file_name}: {e}")
