import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

wav_path = "birlesmis_cikti.wav"

# --- PNG yolunu burada tanımlıyoruz ---
# .wav uzantısını atıp sonuna .png ekler (Örn: 11010330_1eIsABM6.png)
png_path = os.path.splitext(wav_path)[0] + "_2100hz_" + ".png"

try:
    # Sesi yükle
    y, sr = librosa.load(wav_path, sr=None)

    # Mel-Spektrogram hesapla
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2100)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Resim çizimi
    fig = plt.figure(figsize=(3, 3))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=2100)

    # Kaydet
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)

    # Belleği temizle
    plt.close(fig)
    print(f"Başarılı: {os.path.basename(png_path)}")

except Exception as e:
    print(f"Hata: {os.path.basename(wav_path)} -> {e}")