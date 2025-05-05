# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import numpy as np
import json
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
# Load Dataset from Drive
# dataset_path = "/content/drive/My Drive/bengali-convo.csv"  # Update this path
# data = pd.read_csv(dataset_path)
# Enable GPU memory growth
inputs, tags = [], []
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) available and ready for use.")
    except RuntimeError as e:
        print(e)



with open("/content/drive/My Drive/intents1.json", encoding="utf-8") as file:
    intents = json.load(file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Convert pattern to string if it's not already
        if not isinstance(pattern, str):
            pattern = str(pattern)
        inputs.append(pattern)
        tags.append(intent['tag'])




print(inputs)

tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(inputs)
x_train = pad_sequences(tokenizer.texts_to_sequences(inputs), padding='post')

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(tags)

# Model parameters
unique_words = len(tokenizer.word_index)
input_shape = x_train.shape[1]
output_length = len(label_encoder.classes_)

# Model definition
model = Sequential([
    Embedding(unique_words + 1, 50, input_length=input_shape),
    Bidirectional(LSTM(10, return_sequences=True)),
    Bidirectional(LSTM(10, return_sequences=True)),
    Dropout(0.5),
    Flatten(),
    Dense(units=10, activation='relu'),
    Dense(units=output_length, activation='softmax')
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs=40, batch_size=64, validation_split=0.2)

# Save the model
model.save("BanglaBot.h5")
