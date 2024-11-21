from UTIL.reconstruct_audio import reconstruct
from datetime import datetime 
import json
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Activation, Embedding, Input

_LOG = ""  # Para generar el log final

# 00 - Configurar RUTA DEL DATASET
dataset_path = "dataset/audio"
transcripts_path = "dataset/transcript"
audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
print("==============   STEP 0 of X  LOAD AUDIO FILES         ==================")

# 01 - OBTENER DURACIÓN MÁXIMA DEL AUDIO
max_duration = 0
for audio_file in audio_files:
    y, sr = librosa.load(os.path.join(dataset_path, audio_file), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    max_duration = max(max_duration, duration)

# Calcular la longitud máxima en frames (200 ms por frame)
FRAME_STEP = 0.2  # Tamaño de cada frame en segundos
MAX_AUDIO_LEN = int(np.ceil(max_duration / FRAME_STEP))
print("==============   STEP 1 of X  LOAD MAX LEN AUDIO       ==================")

def load_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr, max_len):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = mfccs.T
    if mfccs.shape[0] < max_len:
        mfccs = np.pad(mfccs, ((0, max_len - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:max_len, :]
    return mfccs

audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
transcript_files = [f for f in os.listdir(transcripts_path) if f.endswith('.txt')]

_LOG = _LOG + f"TOTAL AUDIOS.WAV DETECTADOS: {len(audio_files)}\n"
_LOG = _LOG + f"AUDIOS.WAV DETECTADOS: \n{audio_files}\n"
_LOG = _LOG + f"TOTAL AUDIOS.TXT DETECTADOS: {len(transcript_files)}\n"
_LOG = _LOG + f"AUDIOS.TXT DETECTADOS: \n{transcript_files}\n"

audio_data = []
transcripts = []

print("==============   STEP 2 of X  PROCESS AUDIO's          ==================")
for audio_file, transcript_file in zip(audio_files, transcript_files):
    y, sr = load_audio_file(os.path.join(dataset_path, audio_file))
    mfccs = extract_features(y, sr, max_len=MAX_AUDIO_LEN)
    try:
        reconstruct(mfccs, sr, audio_file)
    except ValueError as e:
        print(f"Error al procesar el audio: {e}")
    audio_data.append(mfccs)
    
    with open(os.path.join(transcripts_path, transcript_file), 'r', encoding="UTF-8") as f:
        transcript = f.read().strip()
        transcripts.append(transcript)

audio_data = np.array(audio_data)
transcripts = np.array(transcripts)
print("==============   STEP 3 of X  NEURAL INPUT INFORMATION ==================")
print(f"Audio data shape: {audio_data.shape}")
print(f"Transcripts shape: {transcripts.shape}")

_LOG = _LOG + f"Forma de los datos de audio: {audio_data.shape}\n" 
_LOG = _LOG + f"Forma de las transcripciones: {transcripts.shape}\n"

characters = set("".join(transcripts))
with open("DATA/characters.json", "w", encoding="UTF-8") as f:
    f.write(json.dumps(list(characters)))
char_to_index = {ch: idx for idx, ch in enumerate(characters)}
with open("DATA/char_to_index.json", "w", encoding="UTF-8") as f:
    f.write(json.dumps(char_to_index))

def encode_transcript(transcript):
    return [char_to_index[ch] for ch in transcript]

encoded_transcripts = [encode_transcript(transcript) for transcript in transcripts]

max_len_transcripts = MAX_AUDIO_LEN
padded_transcripts = pad_sequences(encoded_transcripts, maxlen=max_len_transcripts, padding='post')

_LOG = _LOG + f"Forma de las transcripciones codificadas y rellenas: {padded_transcripts.shape}\n"

padded_transcripts = np.expand_dims(padded_transcripts, axis=-1)

model = Sequential()
model.add(Input(shape=(MAX_AUDIO_LEN, 13)))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(len(char_to_index))))
model.add(Activation('softmax'))

_layer_sizes = []
for layer in model.layers:
    if isinstance(layer, TimeDistributed) and isinstance(layer.layer, Dense):
        _layer_sizes.append(layer.layer.units)
    elif isinstance(layer, Dense):
        _layer_sizes.append(layer.units)
_LOG = _LOG + f"CAPAS DEL MODELO: {str(_layer_sizes)}\n"

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(audio_data, padded_transcripts, epochs=50, batch_size=32, validation_split=0.2)

now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d-%H.%M")
_output_model_filename = f"DATA/model-{formatted_date}.keras"
model.save(_output_model_filename)
_LOG = _LOG + f"MODELO GUARDADO: {_output_model_filename}\n"
print(f"MODELO GUARDADO: {_output_model_filename}")

with open("logs.log", "a", encoding="UTF-8") as f:
    f.write(_LOG)
