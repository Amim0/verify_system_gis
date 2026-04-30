import sys
import subprocess
from datetime import datetime
from pathlib import Path

class PipelineRunner:

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[1]

        self.ART_DIR = self.BASE_DIR / "art"
        self.ART_DIR.mkdir(exist_ok=True)

        self.name = None
        self.mode = None
        self.threshold = None

        self.log_file = None

    def collect_input(self):

        while True:
            name = input("Введите имя модели: ").strip()
            if not name:
                print("Имя не может быть пустым")
                continue
            if any(c in name for c in r'\/:*?"<>|'):
                print("Недопустимые символы в имени")
                continue
            self.name = name
            break

        while True:
            mode = input("Режим экспорта (clean/corrupt): ").strip()
            if mode in ("clean", "corrupt"):
                self.mode = mode
                break
            print("Введите clean или corrupt")

        while True:
            try:
                threshold = float(input("Порог воспроизводимости (0–1, default 0.75): ").strip())
                if 0 <= threshold <= 1:
                    self.threshold = threshold
                    break
                print("Введите число от 0 до 1")
            except ValueError:
                print("Ошибка ввода")

    def init_log(self):

        log_path = self.ART_DIR / f"report_{self.name}.txt"
        self.log_file = open(log_path, "w", encoding="utf-8")

        self.log("=" * 55)
        self.log("  ОТЧЁТ ВЕРИФИКАЦИИ МЛ-МОДЕЛИ")
        self.log("=" * 55)
        self.log(f"  Дата запуска : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"  Модель       : {self.name}")
        self.log(f"  Режим        : {self.mode}")
        self.log(f"  Порог        : {self.threshold:.2f}")
        self.log(f"  Формат       : ModelKit (.kit)")
        self.log("=" * 55)

    def log(self, text: str = ""):
        print(text)
        self.log_file.write(text + "\n")
        self.log_file.flush()

    def log_header(self, title: str):
        self.log()
        self.log("=" * 55)
        self.log(f"  {title}")
        self.log("=" * 55)

    def run_step(self, step_num: int, title: str, script: str, stdin_input: str = ""):

        self.log_header(f"ШАГ {step_num}: {title}")

        script_path = self.BASE_DIR / "src" / script

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding="cp1251",
            errors="ignore",
            input=stdin_input
        )

        if result.stdout:
            for line in result.stdout.splitlines():
                self.log("  " + line)

        if result.stderr:
            self.log()
            self.log("  [ ОШИБКИ ]")
            for line in result.stderr.splitlines():
                self.log("  " + line)

        self.log()

        if result.returncode != 0:
            self.log(f"  Шаг завершился с ошибкой (code {result.returncode})")
            self.log("  Остановка.")
            self.log_file.close()
            sys.exit(result.returncode)
        else:
            self.log(f"  Шаг выполнен успешно")

    def run(self):

        self.collect_input()
        self.init_log()

        self.run_step(1, "ОБУЧЕНИЕ",
                      "train_model.py",
                      stdin_input=self.name + "\n")

        self.run_step(2, "ЭТАЛОННАЯ ВЫБОРКА",
                      "reference.py",
                      stdin_input=self.name + "\n")

        self.run_step(3, "ЭКСПОРТ → ModelKit + RSA подпись",
                      "export.py",
                      stdin_input=f"{self.name}\n{self.mode}\n")

        self.run_step(4, "ВЕРИФИКАЦИЯ",
                      "verify.py",
                      stdin_input=f"{self.name}\n{self.threshold}\n")

        self.log()
        self.log("=" * 55)
        self.log(f"  Готово. Отчёт сохранён: {self.ART_DIR / f'report_{self.name}.txt'}")
        self.log("=" * 55)

        self.log_file.close()
if __name__ == "__main__":
    app = PipelineRunner()
    app.run()
