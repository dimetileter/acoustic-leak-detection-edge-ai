import time
import random
import os
import pandas as pd

from Crypto_Manager import CryptoManager
from Water_Leak_IoT_Agent import WaterLeakIoTAgent

def mock_lstm_prediction(cnn_buffer, ground_truth):
    """
    Simülasyon amaçlı sahte bir LSTM kararı üretir.
    Gerçekte bu buffer dizisi LSTM modeline verilecek.
    Mantığı simüle etmek için, eğer gerçekte sızıntı varsa (ve tutarlıysa) %90 üzeri değer döndürüyoruz.
    """
    if ground_truth == 1:
        # Geçici bir gürültü mü yoksa gerçek sızıntı mı anlamak için 
        # (Şu anki CSV simülasyonunda gerçeği yansıtacak şekilde %95 sızıntı diyelim)
        return round(random.uniform(0.90, 0.99), 4)
    else:
        return round(random.uniform(0.01, 0.15), 4)

def start_simulation(interval_seconds=1):
    agent = WaterLeakIoTAgent("SNSR_KNY_01")
    crypto = CryptoManager("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")

    csv_path = "dataset/water_leak_detection_1000_rows.csv"
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)

    print(f"--- Hibrit AI Simülasyonu Başlatıldı (Sliding Window: 5 Sn) ---")
    print(f"Sensör Veri Seti Okunuyor: {len(df)} Satır.\n")

    for index, row in df.iterrows():
        # 1. CSV'den Çevresel Değerleri ve Etiketi Al
        current_pressure = row['Pressure (bar)']
        current_temp = row['Temperature (°C)']
        is_leak = int(row['Leak Status'])

        # 2. Akustik Veriyi Seç ve CNN Öznitelik Vektörünü (Göz) Çıkar
        audio_folder = "dataset/leak" if is_leak == 1 else "dataset/no-leak"
        wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')] if os.path.exists(audio_folder) else []
        
        if wav_files:
            random_wav = os.path.join(audio_folder, random.choice(wav_files))
            cnn_feature = agent.extract_cnn_features(random_wav)
        else:
            cnn_feature = agent.extract_cnn_features("dummy.wav")

        # 3. Spektrogram vektörünü tampon belleğe ekle (Kayar Pencere)
        agent.cnn_feature_buffer.append(cnn_feature)

        # 4. Titreşim Verisini Seç
        vib_folder = "dataset/MEMS Accelerometers/Leak" if is_leak == 1 else "dataset/MEMS Accelerometers/No-Leak"
        vib_files = [f for f in os.listdir(vib_folder) if f.endswith('.csv')] if os.path.exists(vib_folder) else []
        
        if vib_files:
            random_vib = os.path.join(vib_folder, random.choice(vib_files))
            vib_feats = agent.extract_vibration_features(random_vib)
        else:
            vib_feats = {
                "std_dev": round(random.uniform(1.0, 3.0) if is_leak else random.uniform(0.1, 0.5), 4),
                "mean_acc": round(random.uniform(0.2, 0.8) if is_leak else random.uniform(-0.1, 0.1), 4)
            }

        # 5. LSTM (Beyin) Zaman Analizi Yap ve Karar Ver
        if len(agent.cnn_feature_buffer) < 5:
            print(f"Pencere doluyor... ({len(agent.cnn_feature_buffer)}/5)")
        else:
            # Tam 5 saniyelik veri birikti. LSTM Tahmini:
            lstm_prob = mock_lstm_prediction(list(agent.cnn_feature_buffer), is_leak)
            
            # JSON Paketi Oluştur ve Uçtan şifrele
            payload = agent.create_secure_payloads(lstm_prob, vib_feats, current_pressure, current_temp, is_leak)
            encrypted_payload = crypto.encrypt(str(payload))

            # Yapay Zeka Kararı
            ai_decision = "!!! SIZINTI TESPİT EDİLDİ !!!" if lstm_prob > 0.90 else "Sızıntı Yok (Normal/Gürültü)"

            print(f"\n[YAPAY ZEKA]: {ai_decision} | GERÇEK DURUM: {'SIZINTI' if is_leak else 'NORMAL'}")
            print(f"Çevresel Veriler -> Basınç: {current_pressure:.2f} Bar, Sıcaklık: {current_temp:.1f}°C")
            print(f"LSTM Güveni: %{lstm_prob*100:.1f} | Şifreli Payload: {encrypted_payload[:40]}...")

        # Gerçek zamanı simüle etmek için bekle (1 sn adım)
        time.sleep(interval_seconds)

if __name__ == "__main__":
    start_simulation()