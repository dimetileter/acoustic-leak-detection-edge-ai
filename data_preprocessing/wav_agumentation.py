import os
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


def augment_audio_dataset(leak_dir, no_leak_dir, target_dir, augmentations_per_file=10):
    """
    Belirtilen 'leak' ve 'no_leak' klasörlerindeki WAV dosyalarını alır,
    ses efektleriyle çoğaltır ve hedef klasörde aynı isimli alt klasörlere kaydeder.
    """

    # 1. Pipeline'ı Kur
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        Shift(p=0.5)
    ])

    # 2. Kaynak Yolları Sözlükte Eşleştir
    tasks = {
        "leak": leak_dir,
        "no_leak": no_leak_dir
    }

    # Hedef ana klasörü oluştur
    os.makedirs(target_dir, exist_ok=True)

    for label, source_path in tasks.items():
        # Hedefte leak / no_leak alt klasörlerini oluştur
        current_target_dir = os.path.join(target_dir, label)
        os.makedirs(current_target_dir, exist_ok=True)

        if not os.path.exists(source_path):
            print(f"Uyarı: Kaynak klasör bulunamadı, atlanıyor -> {source_path}")
            continue

        print(f"\n[{label.upper()}] sınıfı işleniyor... (Kaynak: {source_path})")

        # Klasördeki sadece WAV dosyalarını bul
        wav_files = [f for f in os.listdir(source_path) if f.endswith('.wav')]

        for file_name in wav_files:
            file_path = os.path.join(source_path, file_name)

            try:
                # Sesi orijinal frekansıyla yükle
                samples, sample_rate = librosa.load(file_path, sr=None)

                # A) Orijinal dosyayı da eğitim setine dahil etmek için kaydet
                sf.write(os.path.join(current_target_dir, f"orig_{file_name}"), samples, sample_rate)

                # B) Belirlenen sayı kadar farklı varyasyon üret
                for i in range(augmentations_per_file):
                    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

                    new_file_name = f"aug_{i}_{file_name}"
                    sf.write(os.path.join(current_target_dir, new_file_name), augmented_samples, sample_rate)

            except Exception as e:
                print(f"Hata ({file_name}): {e}")

        print(
            f"Bitti: {label} klasörüne (1 orijinal + {augmentations_per_file} üretilmiş) * {len(wav_files)} dosya kaydedildi.")



if __name__ == "__main__":

    # Mevcut dosyanın bulunduğu dizin (data_preprocessing)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Proje kök dizini (data_preprocessing'in bir üstü)
    PROJECT_ROOT = os.path.dirname(BASE_DIR)

    LEAK_KLASORU = os.path.join(PROJECT_ROOT, "dataset", "sound_datasets_for_test", "leak")
    NO_LEAK_KLASORU = os.path.join(PROJECT_ROOT, "dataset", "sound_datasets_for_test", "no-leak")

    # Çoğaltılmış seslerin gideceği yeni klasör
    HEDEF_KLASOR = os.path.join(PROJECT_ROOT, "dataset", "sound_datasets_for_test", "audio_agumentation_only_100")

    # Her bir dosyadan 15 tane üret, 70 resim -> 1050 resşm
    augment_audio_dataset(
        leak_dir=LEAK_KLASORU,
        no_leak_dir=NO_LEAK_KLASORU,
        target_dir=HEDEF_KLASOR,
        augmentations_per_file=15
    )