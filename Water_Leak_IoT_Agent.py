import librosa
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd

class WaterLeakIoTAgent:

    def __init__(self, sensor_id):
        self.sensor_id = sensor_id

    # Ses verilerini çıkar
    def extract_acoustic_features(self, wav_path):
        """
        Sensörün aldığı sesi işler ve ses dosyasından özellikler çıkarır
        :param wav_path:
        :return: dict
        """

        y, sr = librosa.load(wav_path)
        # Sensörün aldığı ses şiddetini hesapla
        rms = np.sqrt(np.mean(y**2))
        # Hangi frekansta yoğunluk olduğuna bak
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        # Sözlük olarak işlenen ses verisini dön
        return {"rms": float(rms), "centroid": float(spectral_centroid)}

    # Titreşim verilerini çıkar
    def extract_vibration_features(self, file_path):
        """CSV veya Excel titreşim verisinden özellik çıkarımı yapar."""
        # Dosya uzantısına göre uygun okuma metodunu seç
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # Excel (.xlsx veya .xls) dosyalarını oku
            df = pd.read_excel(file_path)

        # 'Acceleration value' sütunundaki veriyi kullan
        # Sinyalin kararlılığını ölçmek için Standart Sapma hesapla
        vibration_std = df['Acceleration value'].std()
        vibration_mean = df['Acceleration value'].mean()

        return {
            "std_dev": float(vibration_std),
            "mean_acc": float(vibration_mean)
        }

    # JSON verisini oluştur
    def create_secure_payloads(self, audio_data, vib_data):
        """
        Tüm verileri JSON formatına çevirir ve şifrelemeye hazır hale getirir
        :param audio_data:
        :param v:
        :param b_data:
        :return: dict
        """
        payload = {
            "sensor_id": self.sensor_id,
            "timestamp": datetime.now().isoformat(),
            "measurements": {
                "acoustic": audio_data,
                "vibration": vib_data,
                "simulated_pressure": round(random.uniform(3.0, 5.0), 2)
            }
        }

        return payload