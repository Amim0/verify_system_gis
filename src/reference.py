import pickle
from pathlib import Path

class ReferenceApp:
    """
    Формирование эталонных образцов генерации текста
    для последующей проверки воспроизводимости модели.

    Модель генерирует текст, результат сохраняется как эталон
    """
    REF_STARTS = [0, 100, 500, 1000, 1500]
    SEED_LEN = 40
    GEN_LENGTH = 200
    TEMPERATURE = 0.8

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[1]

        self.model_name = input("Имя модели: ").strip()

        self.model_path = self.BASE_DIR / "art" / f"{self.model_name}.pickle"
        self.ref_path = self.BASE_DIR / "art" / f"{self.model_name}_ref.pickle"

        self.model = None
        self.refs = None

    # загрузка модели
    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

    # построение эталона
    def build_reference(self, model, text):
        refs = []

        for start in self.REF_STARTS:
            seed = text[start:start + self.SEED_LEN]

            output = model.generate_text(
                seed,
                length=self.GEN_LENGTH,
                temperature=self.TEMPERATURE
            )

            refs.append({
                "start": start,
                "seed": seed,
                "output": output
            })

        return refs

    #сохранение
    def save_reference(self, refs):
        with open(self.ref_path, "wb") as f:
            pickle.dump(refs, f)

    def run(self):

        self.load_model()

        text = self.model.text

        self.refs = self.build_reference(self.model, text)

        self.save_reference(self.refs)

        print(f"Reference сохранён: {self.ref_path}")

if __name__ == "__main__":
    app = ReferenceApp()
    app.run()