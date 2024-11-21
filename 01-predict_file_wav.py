"""
FelipedelosH
"""
from UTIL.reconstruct_audio import reconstruct
from datetime import datetime
import json
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# LOAD MODEL
model_filename = 'DATA/model-2024-11-21-12.46.keras'
model = load_model(model_filename)

# characters
with open("DATA/char_to_index.json", "r", encoding="UTF-8") as f:
    char_to_index = json.loads(f.read())

index_to_char = {idx: ch for ch, idx in char_to_index.items()}

AUDIO_FILENAME = "DATA/holamundo.wav"

# Proccess Audio
def load_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr, max_len=1732):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = librosa.util.fix_length(mfccs, size=max_len, axis=1)
    reconstruct(mfccs, sr, "holamundo.wav")
    return mfccs

# Cargar y procesar el archivo de audio
y, sr = load_audio_file(AUDIO_FILENAME)
mfccs = extract_features(y, sr)
mfccs = np.expand_dims(mfccs, axis=0)
print(f"mfccs Dimensions: {mfccs.shape}")

# Decodificar las predicciones
def decode_predictions(preds):
    text = ''.join([index_to_char[np.argmax(pred)] for pred in preds[0]])
    return text

# Hacer la predicción y decodificar
predicted_output = model.predict(mfccs)
predicted_text = decode_predictions(predicted_output)

print(f"Texto Predicho: {predicted_text}")
