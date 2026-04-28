from src.model_text_gen import TextGenerator
import pickle
import sys
from pathlib import Path

class TextTrainingApp:
    """
    Обучение LSTM-модели на текстовом корпусе и сохранение результата.
    Этапы:
    - подготовка данных
    - обучение модели
    - сохранение
    - демонстрация генерации
    """

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[1]

        self.name = sys.stdin.readline().strip()

        self.save_path = self.BASE_DIR / "art" / f"{self.name}.pickle"
        self.data_path = self.BASE_DIR / "data" / "text.txt"

        self.gen = None

    # ===== обучение =====
    def train_model(self, gen, x, y, epochs=20, batch_size=128):
        return gen.model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=2
        )

    # ===== основной процесс =====
    def run(self):

        self.gen = TextGenerator(str(self.data_path))

        if not self.gen.load_text():
            return

        self.gen.create_dictionary()
        self.gen.build_model()

        x, y = self.gen.vectorize_text()

        self.train_model(self.gen, x, y, epochs=20, batch_size=128)

        with open(self.save_path, 'wb') as f:
            pickle.dump(self.gen, f)

        # демонстрация (как было)
        seed = self.gen.text[:self.gen.sequence_length]

        print("\n--- DEMO ---")
        print(self.gen.generate_text(seed, length=300, temperature=0.8))


if __name__ == "__main__":
    app = TextTrainingApp()
    app.run()