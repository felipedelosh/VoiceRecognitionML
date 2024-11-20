"""
FelipedelosH
"""
from datetime import datetime
import json
import os
import librosa
import numpy as np
import pyaudio
import wave
from tensorflow.keras.models import load_model

# Cargar el modelo
model_filename = 'DATA/model-2024-11-20-18.25.keras'
model = load_model(model_filename)

# Cargar characters
with open("DATA/characters.json", "r", encoding="UTF-8") as f:
    characters = json.loads(f.read())
# Cargar char_to_index
with open("DATA/char_to_index.json", "r", encoding="UTF-8") as f:
    char_to_index = json.loads(f.read())
# Crear index_to_char
index_to_char = {idx: ch for ch, idx in char_to_index.items()}

# Parámetros de grabación
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5  # Ajusta la duración de la grabación en segundos
WAVE_OUTPUT_FILENAME = "DATA/output.wav"

# Iniciar PyAudio
audio = pyaudio.PyAudio()

# Iniciar la grabación
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Grabando...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Grabación completa")

# Detener la grabación
stream.stop_stream()
stream.close()
audio.terminate()

# Guardar la grabación en un archivo WAV
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# Procesar el archivo de audio grabado
def load_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = librosa.util.fix_length(mfccs, size=max_len, axis=1)
    return mfccs

# Cargar y procesar el archivo de audio
y, sr = load_audio_file(WAVE_OUTPUT_FILENAME)
mfccs = extract_features(y, sr)
mfccs = np.expand_dims(mfccs, axis=0)

# Decodificar las predicciones
def decode_predictions(preds):
    text = ''.join([index_to_char[np.argmax(pred)] for pred in preds[0]])
    return text

# Hacer la predicción y decodificar
predicted_output = model.predict(mfccs)
predicted_text = decode_predictions(predicted_output)

print(f"Texto Predicho: {predicted_text}")
