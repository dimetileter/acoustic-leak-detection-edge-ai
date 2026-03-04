from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64


class CryptoManager:
    def __init__(self, key_hex):
        # AES-256 için 32 byte (64 karakter hex) anahtar gerekir
        self.key = bytes.fromhex(key_hex)

    # Veriyi şifrele
    def encrypt(self, data_string):
        # Rastgele Initialization Vector (IV) oluştur
        cipher = AES.new(self.key, AES.MODE_CBC)
        iv = cipher.iv

        encrypted_bytes = cipher.encrypt(pad(data_string.encode(), AES.block_size))

        # IV ve şifreli veriyi birleştirip base64 formatında döndür
        return base64.b64encode(iv + encrypted_bytes).decode('utf-8')

# Test
MASTER_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

crypto = CryptoManager(MASTER_KEY)
encrypted_msg = crypto.encrypt('{"sensor_id": "SNSR-01", "status": "leak"}')
print(f"Şifreli Paket: {encrypted_msg}")