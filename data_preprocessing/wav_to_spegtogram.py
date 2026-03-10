import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor


# 1. TEK BİR DOSYAYI İŞLEYEN FONKSİYON (Worker)
def process_single_wav(task_args):
    """
    Bu fonksiyon her bir çekirdek tarafından bağımsız olarak çalıştırılacak.
    task_args: (wav_path, png_path) şeklinde bir tuple.
    """
    wav_path, png_path = task_args

    try:
        # Sesi yükle (sr=None orijinal hızda yükler)
        y, sr = librosa.load(wav_path, sr=None)

        # Mel-Spektrogram hesapla
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Resim çizimi
        fig = plt.figure(figsize=(3, 3))
        plt.axis('off')
        librosa.display.specshow(S_dB, sr=sr, fmax=8000)

        # Kaydet
        plt.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)

        # Belleği temizle
        plt.close(fig)
        return f"Başarılı: {os.path.basename(png_path)}"

    except Exception as e:
        return f"Hata: {os.path.basename(wav_path)} -> {e}"


# 2. ANA YÖNETİCİ FONKSİYON
def create_spectrograms_parallel(leak_dir, no_leak_dir, target_dir):
    directories = {"leak": leak_dir, "no_leak": no_leak_dir}
    all_tasks = []

    # Dosya yollarını listede topla
    for label, source_path in directories.items():
        current_target_dir = os.path.join(target_dir, label)
        # Hedef klasör yoksa oluştur
        os.makedirs(current_target_dir, exist_ok=True)

        # Kalsördeki ses dosyalarını tek tek gez
        for file_name in os.listdir(source_path):

            # Wav dosyalarının yolunu al ve png uzantılı resim isimlerini oluştur
            if file_name.endswith('.wav'):
                wav_path = os.path.join(source_path, file_name)
                png_path = os.path.join(current_target_dir, file_name.replace('.wav', '.png'))

                # İşlenecek sesleri listeye ekle
                all_tasks.append((wav_path, png_path))

    print(f"-> Toplam {len(all_tasks)} dosya bulundu.")
    print(f"-> Resim çevirme işlemi başlıyor...")

    # 3. process_single_wav fonksiyonunu paralel olarak işle
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_wav, all_tasks))

    # Sonuçları raporla
    for res in results:
        if "Hata" in res:
            print(res)

    print("\n[TAMAMLANDI] Tüm veri seti başarıyla resme çevrildi.")


# WINDOWS İÇİN KRİTİK NOKTA:
if __name__ == "__main__":
    leak_dir = "dataset/sound_datasets_for_test/audio_agumentation_only_40/leak"
    no_leak_dir = "dataset/sound_datasets_for_test/audio_agumentation_only_40/no_leak"
    target_dir = "dataset/image_datasets/spectrogram_pool_augmented_from_only_40"

    create_spectrograms_parallel(leak_dir, no_leak_dir, target_dir)