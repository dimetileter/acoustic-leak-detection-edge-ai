import time
import random
import os

from Crypto_Manager import CryptoManager
from Water_Leak_IoT_Agent import WaterLeakIoTAgent


# Daha önce yazdığımız sınıfları import ettiğini varsayalım
# from virtual_IoT_agent import WaterLeakIoTAgent
# from security_manager import CryptoManager

def start_simulation(interval_seconds=5):
    agent = WaterLeakIoTAgent("SNSR_KNY_01")
    crypto = CryptoManager("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")

    print(f"--- {interval_seconds} saniye aralıklarla veri akışı başlatıldı ---")

    while True:
        # 1. Senaryoyu Rastgele Seç (%20 ihtimalle sızıntı olsun)
        is_leak = random.random() < 0.2

        # 2. Rastgele Dosya Yollarını Belirle
        # (Gerçek klasör yollarını buraya eklemelisin)
        folder = "dataset/Hydrophones/Leak" if is_leak else "dataset/Hydrophones/No-Leak"
        random_wav = os.path.join(folder, random.choice(os.listdir(folder)))

        # 3. Veriyi İşle (Kenar Hesaplama)
        audio_feats = agent.extract_acoustic_features(random_wav)
        # Excel verisini de benzer şekilde rastgele seçip işleyebilirsin
        vib_feats = agent.extract_vibration_features("dataset/MEMS Accelerometers/No-Leak/1. A50_11_25_2020.xlsx")

        # 4. JSON Paketi Oluştur ve ŞİFRELE
        payload = agent.create_secure_payload(audio_feats, vib_feats)
        encrypted_payload = crypto.encrypt(str(payload))

        # 5. Ekrana Yazdır (Veya Firebase'e Gönder)
        status = "!!! SIZINTI !!!" if is_leak else "NORMAL"
        print(f"[{status}] - Zaman: {payload['timestamp']}")
        print(f"Gönderilen Şifreli Paket: {encrypted_payload[:50]}...")

        # Belirlenen süre kadar bekle
        time.sleep(interval_seconds)


if __name__ == "__main__":
    start_simulation()