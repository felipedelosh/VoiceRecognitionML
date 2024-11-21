"""
FelipedelosH
"""
from UTIL.reconstruct_audio import reconstruct
from datetime import datetime 
import json
import os
import librosa
import soundfile as sf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Embedding, Input

_LOG = ""

# 00 Configure path folder
dataset_path = "dataset/audio"
transcripts_path = "dataset/transcript"
audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
print("==============   STEP 0 of X  LOAD AUDIO FILES         ==================")
_txt = f"TOTAL AUDIOS: {len(audio_files)}"
print(_txt)
_LOG = _LOG + _txt + "\n"


# 01 - GET MAX DURATION OF AUDIO.WAV
max_duration = 0
for audio_file in audio_files:
    y, sr = librosa.load(os.path.join(dataset_path, audio_file), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    max_duration = max(max_duration, duration)

# Calcular la longitud máxima en frames (200 ms por frame)
FRAME_STEP = 0.2  # Tamaño de cada frame en segundos
MAX_AUDIO_LEN = int(np.ceil(max_duration / FRAME_STEP))
print("==============   STEP 1 of X  LOAD MAX LEN AUDIO       ==================")
_txt = f"MAX LEN AUDIOS (SEG): {max_duration}"
print(_txt)
_LOG = _LOG + _txt + "\n"


def load_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr, max_len=100):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = librosa.util.fix_length(mfccs, size=max_len, axis=1)
    return mfccs


audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
transcript_files = [f for f in os.listdir(transcripts_path) if f.endswith('.txt')]

_LOG = _LOG + f"TOTAL AUDIOS.WAV DETECTED: {len(audio_files)}\n"
_LOG = _LOG + f"AUDIOS.WAV DETECTED: \n{audio_files}\n"
_LOG = _LOG + f"TOTAL AUDIOS.TXT DETECTED: {len(transcript_files)}\n"
_LOG = _LOG + f"AUDIOS.TXT DETECTED: \n{transcript_files}\n"

audio_data = []
transcripts = []


print("==============   STEP 2 of X  PROCESS AUDIO's          ==================")
for audio_file, transcript_file in zip(audio_files, transcript_files):
    y, sr = load_audio_file(os.path.join(dataset_path, audio_file))
    mfccs = extract_features(y, sr)
    # Save in TEST FOLDER
    reconstruct(mfccs, sr, audio_file)
    # END TO SAVE IN FOLDER TEXT
    audio_data.append(mfccs)
    
    with open(os.path.join(transcripts_path, transcript_file), 'r', encoding="UTF-8") as f:
        transcript = f.read().strip()
        transcripts.append(transcript)

audio_data = np.array(audio_data)
_shape_audio_input = audio_data.shape
_model_n_mfcc = _shape_audio_input[1]
_model_max_len = _shape_audio_input[2]
transcripts = np.array(transcripts)
print("==============   STEP 3 of X  NEURAL INPUT INFORMATION ==================")
print(f"Audio data shape: {_shape_audio_input}")
print(f"Model INPUT: {_model_n_mfcc, _model_max_len}")
print(f"Transcripts shape: {transcripts.shape}")


_LOG = _LOG + f"Audio data shape: {audio_data.shape}\n" 
_LOG = _LOG + f"Transcripts shape: {transcripts.shape}\n" 


# Preprocesing
characters = set("".join(transcripts))
with open("DATA/characters.json", "w", encoding="UTF-8") as f:
    f.write(json.dumps(list(characters)))
char_to_index = {ch: idx for idx, ch in enumerate(characters)}
with open("DATA/char_to_index.json", "w", encoding="UTF-8") as f:
    f.write(json.dumps(char_to_index))

def encode_transcript(transcript):
    return [char_to_index[ch] for ch in transcript]

encoded_transcripts = [encode_transcript(transcript) for transcript in transcripts]

# Padding de las secuencias
max_len_transcripts = 13  # Ajustar a la misma longitud que la salida del modelo
padded_transcripts = pad_sequences(encoded_transcripts, maxlen=max_len_transcripts, padding='post')

_LOG = _LOG + f"Encoded and padded transcripts shape: {padded_transcripts.shape}\n" 

# Ajustar las dimensiones de las transcripciones para que coincidan con las salidas del modelo
padded_transcripts = np.expand_dims(padded_transcripts, axis=-1)

# Definir el modelo
model = Sequential()
model.add(Input(shape=(13, 100)))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(len(characters))))
model.add(Activation('softmax'))

# READ LAYERS ONLY TO SAVE IN LOG
_layer_sizes = []
for layer in model.layers:
    if isinstance(layer, TimeDistributed) and isinstance(layer.layer, Dense):
        _layer_sizes.append(layer.layer.units)
    elif isinstance(layer, Dense):
        _layer_sizes.append(layer.units)
_LOG = _LOG + f"LAYERS OF MODEL: {str(_layer_sizes)}\n"

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train Model
model.fit(audio_data, padded_transcripts, epochs=50, batch_size=32, validation_split=0.2)

# Save model
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d-%H.%M")
_output_model_filename = f"DATA/model-{formatted_date}.keras"
model.save(_output_model_filename)
_LOG = _LOG + f"SAVE MODEL: {_output_model_filename}\n"
print(f"SAVE MODEL: {_output_model_filename}")


# SAVE LOGS
with open("logs.log", "w", encoding="UTF-8") as f:
    f.write(_LOG)
