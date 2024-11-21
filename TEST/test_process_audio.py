"""
FelipedelosH
"""
import librosa
import numpy as np
import soundfile as sf


def load_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=256)
    mfccs = librosa.util.fix_length(mfccs, size=max_len, axis=1)
    return mfccs

def mfcc_to_audio(mfccs, sr):
    # Normalmente necesitas el espectrograma de magnitud para hacer la inversión
    D = librosa.feature.inverse.mfcc_to_audio(mfccs, sr=sr)
    return D

# Ruta del archivo de audio
AUDIO_FILENAME = 'holamundo.wav'
OUTPUT_FILENAME = 'output_processed.wav'

# Cargar y procesar el archivo de audio
y, sr = load_audio_file(AUDIO_FILENAME)
mfccs = extract_features(y, sr, max_len=100)

# Invertir las características MFCC a señal de audio
y_reconstructed = mfcc_to_audio(mfccs, sr)

# Guardar el audio resultante
sf.write(OUTPUT_FILENAME, y_reconstructed, sr)

print(f"Audio procesado guardado en: {OUTPUT_FILENAME}")
