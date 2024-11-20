"""
FelipedelosH
"""
from datetime import datetime
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Embedding, Input

_LOG = ""

# 00 - Procesing AUDIOS
def load_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = librosa.util.fix_length(mfccs, size=max_len, axis=1)
    return mfccs

dataset_path = "dataset/audio"
transcripts_path = "dataset/transcript"

audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
transcript_files = [f for f in os.listdir(transcripts_path) if f.endswith('.txt')]

_LOG = _LOG + f"TOTAL AUDIOS.WAV DETECTED: {len(audio_files)}\n"
_LOG = _LOG + f"AUDIOS.WAV DETECTED: \n{audio_files}\n"
_LOG = _LOG + f"TOTAL AUDIOS.TXT DETECTED: {len(transcript_files)}\n"
_LOG = _LOG + f"AUDIOS.TXT DETECTED: \n{transcript_files}\n"

audio_data = []
transcripts = []

for audio_file, transcript_file in zip(audio_files, transcript_files):
    y, sr = load_audio_file(os.path.join(dataset_path, audio_file))
    mfccs = extract_features(y, sr)
    audio_data.append(mfccs)
    
    with open(os.path.join(transcripts_path, transcript_file), 'r', encoding="UTF-8") as f:
        transcript = f.read().strip()
        transcripts.append(transcript)

audio_data = np.array(audio_data)
transcripts = np.array(transcripts)

print("Audio data shape:", audio_data.shape)
print("Transcripts shape:", transcripts.shape)

# Preprocesar las transcripciones
characters = set("".join(transcripts))
char_to_index = {ch: idx for idx, ch in enumerate(characters)}

def encode_transcript(transcript):
    return [char_to_index[ch] for ch in transcript]

encoded_transcripts = [encode_transcript(transcript) for transcript in transcripts]

# Padding de las secuencias
max_len_transcripts = 13  # Ajustar a la misma longitud que la salida del modelo
padded_transcripts = pad_sequences(encoded_transcripts, maxlen=max_len_transcripts, padding='post')

print("Encoded and padded transcripts shape:", padded_transcripts.shape)

# Ajustar las dimensiones de las transcripciones para que coincidan con las salidas del modelo
padded_transcripts = np.expand_dims(padded_transcripts, axis=-1)

# Definir el modelo
model = Sequential()
model.add(Input(shape=(13, 100)))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(len(characters))))
model.add(Activation('softmax'))

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train Model
model.fit(audio_data, padded_transcripts, epochs=50, batch_size=32, validation_split=0.2)

# Save model
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d-%H.%M")
_output_model_filename = f"model-{formatted_date}.keras"
model.save(_output_model_filename)
print(f"SAVE MODEL: {_output_model_filename}")


# SAVE LOGS
with open("logs.log", "a", encoding="UTF-8") as f:
    f.write(_LOG)
