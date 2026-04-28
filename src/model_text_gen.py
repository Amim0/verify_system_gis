import os
#убрать логи тензорфлоу для чистого вывода
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)
"""
Задание:
Реализация посимвольного генератора текста на основе рекуррентной нейронной сети.

Принцип работы:
Модель обучается предсказывать следующий символ по предыдущим символам последовательности.

Исходные данные:
Текстовый корпус в формате .txt
"""

class TextGenerator:

    #Класс генерации текста на основе LSTM.

    def __init__(self, filepath):
        self.filepath = filepath
        self.text = None

        self.chars = None
        self.char_indices = None
        self.indices_char = None

        self.model = None

        self.sequence_length = 40
        self.step = 3

    def load_text(self):
        # загрузка текста из файла
        try:
            with io.open(self.filepath, encoding="utf-8") as f:
                self.text = f.read().lower()
            return True
        except Exception as e:
            print(f"Ошибка: {e}")
            return False

    def create_dictionary(self):
        #Формирует словарь уникальных символов.
        self.chars = sorted(list(set(self.text)))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}

    def vectorize_text(self):

        sentences = []
        next_chars = []

        for i in range(0, len(self.text) - self.sequence_length, self.step):
            sentences.append(self.text[i:i + self.sequence_length])
            next_chars.append(self.text[i + self.sequence_length])

        x = np.zeros((len(sentences), self.sequence_length, len(self.chars)), dtype=np.bool_)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool_)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        return x, y

    def build_model(self):
        #архитектура нейронной сети
        self.model = Sequential([
            LSTM(128, return_sequences=True,
                 input_shape=(self.sequence_length, len(self.chars))),
            Dropout(0.2),
            LSTM(128),
            Dropout(0.2),
            Dense(len(self.chars), activation="softmax")
        ])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=0.01),
            metrics=["accuracy"]
        )

    def generate_text(self, seed_text, length=200, temperature=1.0):
        #Генерирует текст на основе обученной модели
        if len(seed_text) < self.sequence_length:
            seed_text = self.text[:self.sequence_length]

        generated = seed_text
        sentence = generated[-self.sequence_length:]

        for _ in range(length):
            x_pred = np.zeros((1, self.sequence_length, len(self.chars)))

            for t, char in enumerate(sentence):
                if char in self.char_indices:
                    x_pred[0, t, self.char_indices[char]] = 1

            preds = self.model.predict(x_pred, verbose=0)[0]

            preds = np.asarray(preds).astype("float64")
            preds = np.log(preds + 1e-8) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)

            next_index = np.argmax(preds)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = generated[-self.sequence_length:]

        return generated

    def get_weights(self):
        # Возвращает веса модели
        return self.model.get_weights()

    def set_weights(self, weights):
        # Устанавливает веса модели
        self.model.set_weights(weights)