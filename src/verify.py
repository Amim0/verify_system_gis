"""
verify.py — проверка ML-модели из .kit-артефакта

Проверки:
  1. Подпись — файл не подменён
  2. Целостность — веса совпадают
  3. Поведение — модель генерирует похожий текст
"""
import pickle
import hashlib
import json
import tarfile
import numpy as np
from pathlib import Path
from sign import SignatureService

# Вся модель и связанные данные хранятся внутри .kit-архива.
# Поэтому при проверке используется только содержимое этого файла,
# без загрузки исходной (оригинальной) модели.

class VerifyApp:

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[1]
        self.name = input("Имя модели: ").strip()
        self.threshold = float(input("Порог (0–1): "))

        self.artifact = None

    def load_kit(self, path: Path) -> dict:
        with tarfile.open(str(path), "r:gz") as tar:
            files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}

        return {
            "manifest": json.loads(files["manifest.json"]),
            "model": pickle.loads(files["model.pkl"]),
            "refs": pickle.loads(files["reference.pkl"]),
        }

    def check_signature(self, manifest: dict) -> bool:
        signer = SignatureService()
        return signer.verify(
            manifest.get("weights_hash", ""),
            manifest.get("signature", "")
        )

    def get_weights_hash(self, model) -> str:
        raw = b"".join(w.tobytes() for w in model.get_weights())
        return hashlib.sha256(raw).hexdigest()

    def check_integrity(self, manifest, model_exp):
        h1 = manifest.get("weights_hash", "")  # хэш оригинала из манифеста
        h2 = self.get_weights_hash(model_exp)  # хэш экспортированной
        return h1 == h2, h1, h2

    def _ngram_jaccard(self, a: str, b: str, n: int) -> float:
        def ngrams(t):
            return set(t[i:i+n] for i in range(len(t) - n + 1))
        A, B = ngrams(a), ngrams(b)
        return len(A & B) / len(A | B) if A and B else 0.0

    def _word_overlap(self, a: str, b: str) -> float:
        wa, wb = set(a.lower().split()), set(b.lower().split())
        return len(wa & wb) / len(wa | wb) if wa and wb else 0.0

    def check_reproducibility(self, model_export, refs):
        scores = []
        details = []

        for i, ref in enumerate(refs):
            seed = ref["seed"]
            output_ref = ref["output"]  # берём сохранённый эталон

            # генерируем только экспортированной моделью
            out_export = model_export.generate_text(seed, 200, 0.8)

            char_score = (
                    0.10 * self._ngram_jaccard(output_ref, out_export, 1) +
                    0.20 * self._ngram_jaccard(output_ref, out_export, 2) +
                    0.30 * self._ngram_jaccard(output_ref, out_export, 3) +
                    0.40 * self._ngram_jaccard(output_ref, out_export, 4)
            )

            total = 0.70 * char_score + 0.30 * self._word_overlap(output_ref, out_export)

            scores.append(total)

            if i < 3:
                details.append({
                    "idx": i + 1,
                    "seed": seed,
                    "out_a": output_ref,  # эталон
                    "out_b": out_export,  # экспорт
                    "score": total
                })

        return float(np.mean(scores)), details

    def print_report(self, name, mode, sig_ok, integrity_ok,
                     h1, h2, repro_score, threshold, samples):

        print("\n" + "─" * 50)
        print(f"МОДЕЛЬ: {name} | режим: {mode}")
        print("─" * 50)

        print(f"Подпись:      {'OK' if sig_ok else 'FAIL'}")
        print(f"Целостность:  {'OK' if integrity_ok else 'FAIL'}")

        print("\nХэши:")
        print(f"  оригинал : {h1[:32]}...")
        print(f"  экспорт  : {h2[:32]}...")

        print("\nПримеры генерации:")

        for s in samples:
            print(f"\n--- Пример {s['idx']} ---")
            print(f"фрагмент: {s['seed'][:50]}...")

            print("\nоригинал:")
            print(s["out_a"][:120].replace("\n", " "), "...")

            print("\nэкспортированная:")
            print(s["out_b"][:120].replace("\n", " "), "...")

            print(f"score: {s['score']:.3f}")

        print("\nВоспроизводимость:")
        print(f"  score = {repro_score:.3f} (порог {threshold})")

        passed = sig_ok and integrity_ok and (repro_score >= threshold)

        print("─" * 50)
        print("Модель корректна" if passed else "Модель не прошла проверку")
        print("─" * 50 + "\n")

        return passed

    def verify(self):
        kit_path = self.BASE_DIR / "art" / f"{self.name}_exported.kit"
        self.artifact = self.load_kit(kit_path)

        model_exp = self.artifact["model"]
        refs = self.artifact["refs"]
        manifest = self.artifact["manifest"]

        mode = manifest.get("export_mode", "?")

        print("Проверка подписи...")
        sig_ok = self.check_signature(manifest)

        print("Проверка целостности...")

        integrity_ok, h1, h2 = self.check_integrity(manifest, model_exp)

        print("Проверка воспроизводимости...")
        repro_score, samples = self.check_reproducibility(model_exp, refs)

        return self.print_report(
            self.name, mode, sig_ok, integrity_ok,
            h1, h2, repro_score, self.threshold, samples
        )

if __name__ == "__main__":
    app = VerifyApp()
    app.verify()
