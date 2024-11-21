"""
FelipedelosH
"""
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Input
import json

# DATASET PATH
dataset_path = "dataset/audio"
transcripts_path = "dataset/transcript"
audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]

# Audio with MAX Duration (TIME)
max_duration = 0
for audio_file in audio_files:
    y, sr = librosa.load(os.path.join(dataset_path, audio_file), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    max_duration = max(max_duration, duration)

# Calcular la longitud máxima en frames (200 ms por frame)
FRAME_STEP = 0.2  # Tamaño de cada frame en segundos
MAX_AUDIO_LEN = int(np.ceil(max_duration / FRAME_STEP))

print(f"Duración máxima de audio: {max_duration:.2f} segundos.")
print(f"Longitud máxima en frames: {MAX_AUDIO_LEN}.")


# Preprocesing data
def extract_features(file_path, max_len=MAX_AUDIO_LEN, frame_step=FRAME_STEP):
    y, sr = librosa.load(file_path, sr=None)
    # Extraer MFCCs (13 características por frame)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(frame_step * sr)).T

    # Ajustar a la longitud máxima
    if len(mfccs) < max_len:
        mfccs = np.pad(mfccs, ((0, max_len - len(mfccs)), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:max_len]
    
    return mfccs

# Extraer las características de todos los audios
audio_data = []
for audio_file in audio_files:
    features = extract_features(os.path.join(dataset_path, audio_file))
    audio_data.append(features)

audio_data = np.array(audio_data)
print("=============================")
print(f"Shape de audio_data: {audio_data.shape}")

# Codificate trancripts
# Leer las transcripciones
transcript_files = [f for f in os.listdir(transcripts_path) if f.endswith('.txt')]

transcripts = []
for transcript_file in transcript_files:
    with open(os.path.join(transcripts_path, transcript_file), 'r', encoding='utf-8') as f:
        transcripts.append(f.read().strip())

# Crear un vocabulario
characters = sorted(set("".join(transcripts)))
char_to_index = {char: idx for idx, char in enumerate(characters)}

# Guardar el mapeo para el futuro
with open("char_to_index.json", "w", encoding="utf-8") as f:
    json.dump(char_to_index, f)

# Convertir las transcripciones a índices
encoded_transcripts = [[char_to_index[char] for char in transcript] for transcript in transcripts]

# Alinear las transcripciones a MAX_AUDIO_LEN
padded_transcripts = pad_sequences(encoded_transcripts, maxlen=MAX_AUDIO_LEN, padding='post')

print(f"Shape de padded_transcripts: {padded_transcripts.shape}")



# Neural network
model = Sequential()
model.add(Input(shape=(MAX_AUDIO_LEN, 13)))  # Longitud máxima, 13 MFCCs
model.add(LSTM(128, return_sequences=True))  # Primera capa LSTM
model.add(TimeDistributed(Dense(len(characters))))  # Salida para cada frame
model.add(Activation('softmax'))  # Activación softmax

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Traint Model
# Ajustar las dimensiones de las etiquetas
padded_transcripts = np.expand_dims(padded_transcripts, axis=-1)

# Entrenar el modelo
model.fit(audio_data, padded_transcripts, epochs=20, batch_size=16, validation_split=0.2)

# Guardar el modelo
model.save("speech_to_text_model.keras")
print("Modelo guardado como speech_to_text_model.keras")

from tensorflow.keras.models import load_model

model = load_model("speech_to_text_model.keras")


# Procesar un nuevo archivo de audio
test_audio_file = "DATA/holamundo.wav"
test_features = extract_features(test_audio_file)
test_features = np.expand_dims(test_features, axis=0)  # Añadir dimensión batch

# Realizar la predicción
prediction = model.predict(test_features)

# Decodificar la predicción
decoded_transcription = ''.join([characters[np.argmax(frame)] for frame in prediction[0]])

with open("DATA/output.txt", "w", encoding="UTF-8") as f:
    f.write(decoded_transcription)

print("LISTO!!!")

