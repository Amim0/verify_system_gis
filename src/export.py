"""
export.py — Экспорт модели в ModelKit-артефакт (.kit).
Важно: хэш и подпись считаются от ОРИГИНАЛЬНЫХ весов,
ДО возможного искажения. Это корректная модель:
  - подпись удостоверяет оригинал
  - corrupt-режим симулирует подмену весов злоумышленником
  - верификация обнаруживает расхождение
Структура .kit (tar.gz):
  manifest.json — метаданные + хэш оригинала + RSA-подпись
  model.pkl — модель (может быть искажена в corrupt-режиме)
  reference.pkl — эталонная выборка
"""
import pickle
import hashlib
import numpy as np
import tarfile
import json
import io
from datetime import datetime
from pathlib import Path
from sign import SignatureService

class ExportApp:

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[1]

        self.name = input().strip()
        self.mode = input().strip()

        self.art_dir = self.BASE_DIR / "art"
        self.kit_path = self.art_dir / f"{self.name}_exported.kit"

        self.model = None
        self.refs = None

    def get_weights_hash(self, model) -> str:
        raw = b"".join(w.tobytes() for w in model.get_weights())
        return hashlib.sha256(raw).hexdigest()

    # archive creation
    def create_kit(self, kit_path, model, refs, metadata):
        model_bytes = pickle.dumps(model)
        refs_bytes = pickle.dumps(refs)
        meta_bytes = json.dumps(metadata, indent=2, ensure_ascii=False).encode()

        def add(tar, filename, data):
            ti = tarfile.TarInfo(filename)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))

        with tarfile.open(kit_path, "w:gz") as tar:
            add(tar, "manifest.json", meta_bytes)
            add(tar, "model.pkl", model_bytes)
            add(tar, "reference.pkl", refs_bytes)

    # ===== load =====
    def load_inputs(self):
        with open(self.art_dir / f"{self.name}.pickle", "rb") as f:
            self.model = pickle.load(f)

        with open(self.art_dir / f"{self.name}_ref.pickle", "rb") as f:
            self.refs = pickle.load(f)

    # ===== export logic =====
    def run(self):
        self.load_inputs()

        weights_hash = self.get_weights_hash(self.model)
        signer = SignatureService()
        signature = signer.sign(weights_hash)

        print(f"  Хэш оригинала : {weights_hash[:32]}...")
        print(f"  Подпись       : создана (RSA-PSS SHA-256)")

        # искажение весов ПОСЛЕ подписи
        if self.mode == "corrupt":
            print("  Режим         : corrupt (веса искажены после подписания)")
            weights = [
                w + np.random.normal(0, 0.3, w.shape)
                for w in self.model.get_weights()
            ]
            self.model.set_weights(weights)
        else:
            print("  Режим         : clean")

        metadata = {
            "name": self.name,
            "export_mode": self.mode,
            "created": datetime.now().isoformat(),
            "weights_hash": weights_hash,
            "signature": signature,
        }

        self.create_kit(self.kit_path, self.model, self.refs, metadata)

        print(f"  Артефакт      : {self.kit_path}")


if __name__ == "__main__":
    app = ExportApp()
    app.run()
