import librosa
import json
import random
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd

class WaterLeakIoTAgent:

    def __init__(self, sensor_id):
        self.sensor_id = sensor_id
        # Zamansal LSTM analizi için 5 saniyelik kayar pencere (sliding window) tampon belleği
        self.cnn_feature_buffer = deque(maxlen=5)

    # Ses verilerini çıkar
    # Ses verilerini CNN öznitelik vektörüne çevir (Simülasyon)
    def extract_cnn_features(self, wav_path):
        """
        Gerçek AI entegrasyonunda bu fonksiyon sesi Mel-Spektrograma çevirir
        ve CNN modeline sokarak 128 boyutlu bir öznitelik vektörü döndürür.
        Şu an model hazır olmadığı için rastgele (ama tutarlı) bir 128D vektör üretiyoruz.
        """
        # CNN'nin ürettiği varsayılan 128 özellikli vektör
        cnn_vector = [round(random.uniform(0.0, 1.0), 4) for _ in range(128)]
        return cnn_vector

    # Titreşim verilerini çıkar
    def extract_vibration_features(self, file_path):
        """CSV veya Excel titreşim verisinden özellik çıkarımı yapar."""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        vibration_std = df['Acceleration value'].std()
        vibration_mean = df['Acceleration value'].mean()

        return {
            "std_dev": float(vibration_std),
            "mean_acc": float(vibration_mean)
        }

    # JSON verisini oluştur
    def create_secure_payloads(self, lstm_prob, vib_data, pressure, temperature, ground_truth):
        """
        Tüm verileri (çevresel + yapay zeka çıkarımları) JSON formatına çevirir
        """
        payload = {
            "sensor_id": self.sensor_id,
            "timestamp": datetime.now().isoformat(),
            "measurements": {
                "lstm_leak_probability": lstm_prob,
                "vibration": vib_data,
                "pressure": float(pressure),
                "temperature": float(temperature)
            },
            "ground_truth_label": int(ground_truth)
        }

        return payload