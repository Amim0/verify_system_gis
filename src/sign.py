"""
sign.py — RSA-PSS подпись (SHA-256, 2048 бит).
Ключи генерируются один раз и хранятся в art/keys/.
"""
import base64
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class SignatureService:

    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[1]

        self.KEYS_DIR = self.BASE_DIR / "art" / "keys"
        self.PRIV_PATH = self.KEYS_DIR / "private.pem"
        self.PUB_PATH = self.KEYS_DIR / "public.pem"

    def _pss(self):
        return padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        )

    # ===== key loading / generation =====
    def load_keys(self):
        if self.PRIV_PATH.exists() and self.PUB_PATH.exists():
            with open(self.PRIV_PATH, "rb") as f:
                priv = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )

            with open(self.PUB_PATH, "rb") as f:
                pub = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )

            return priv, pub

        # генерация ключей при первом запуске
        self.KEYS_DIR.mkdir(parents=True, exist_ok=True)

        priv = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        pub = priv.public_key()

        with open(self.PRIV_PATH, "wb") as f:
            f.write(priv.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption()
            ))

        with open(self.PUB_PATH, "wb") as f:
            f.write(pub.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo
            ))

        print("  Ключи сгенерированы: art/keys/")
        return priv, pub

    # SIGN
    def sign(self, data: str) -> str:
        """Подписывает строку, возвращает base64-подпись."""
        priv, _ = self.load_keys()

        sig = priv.sign(
            data.encode(),
            self._pss(),
            hashes.SHA256()
        )

        return base64.b64encode(sig).decode()

    # ===== VERIFY =====
    def verify(self, data: str, sig_b64: str) -> bool:
        """Проверяет подпись. True если корректна."""
        _, pub = self.load_keys()

        try:
            pub.verify(
                base64.b64decode(sig_b64),
                data.encode(),
                self._pss(),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False