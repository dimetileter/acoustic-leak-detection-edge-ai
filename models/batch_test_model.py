# -*- coding: utf-8 -*-
"""
CNN Modeli Toplu Test Scripti
==============================
spectrogram_pool_augmented_test_not_in_train klasorundeki
tum gorselleri tek tek modele gonderir ve sonuclari raporlar.
Confusion matrix, sinif bazli basari ve genel dogruluk gosterir.
"""

import sys
import pathlib
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

# Turkce font destegi icin
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 1. Yollari Tanimla ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "leak_detection_cnn_only_100(2.0).keras"
DATA_DIR = BASE_DIR / "dataset" / "image_datasets" / "spectrogram_pool_augmented_from_only_40"

# Egitim kodundaki siralama ile ayni: leak=0, no_leak=1
SINIF_ISIMLERI = ["leak", "no_leak"]


# --- 2. Gorsel yukleme fonksiyonu (egitim ile ayni on-isleme) ---
def yukle_ve_isle(dosya_yolu):
    image = tf.io.read_file(str(dosya_yolu))
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0
    return tf.expand_dims(image, axis=0)


# --- 3. Modeli Yukle ---
print("=" * 60)
print("  CNN Model - Toplu Test Raporu")
print("=" * 60)
print(f"\n[*] Model yukleniyor: {MODEL_PATH.name}")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("[+] Model yuklendi!\n")


# --- 4. Tum gorselleri test et ---
sonuclar = []  # (dosya_adi, gercek_sinif, tahmin_sinif, sigmoid, dogru_mu)

for sinif in SINIF_ISIMLERI:
    sinif_klasoru = DATA_DIR / sinif
    dosyalar = sorted(sinif_klasoru.glob("*.png"))
    gercek_index = SINIF_ISIMLERI.index(sinif)

    for dosya in dosyalar:
        image_batch = yukle_ve_isle(dosya)
        sigmoid = model.predict(image_batch, verbose=0)[0][0]
        tahmin_index = 1 if sigmoid >= 0.5 else 0
        tahmin_sinif = SINIF_ISIMLERI[tahmin_index]
        dogru_mu = tahmin_index == gercek_index

        sonuclar.append({
            "dosya": dosya.name,
            "gercek": sinif,
            "tahmin": tahmin_sinif,
            "sigmoid": float(sigmoid),
            "dogru": dogru_mu
        })

        # Her sonucu aninda yazdir
        durum = "OK" if dogru_mu else "XX"
        print(f"  [{durum}] {dosya.name:35s} | Gercek: {sinif:8s} | Tahmin: {tahmin_sinif:8s} | Sigmoid: {sigmoid:.4f}")


# --- 5. Istatistikleri Hesapla ---
toplam = len(sonuclar)
dogru_sayisi = sum(1 for s in sonuclar if s["dogru"])
yanlis_sayisi = toplam - dogru_sayisi
genel_basari = (dogru_sayisi / toplam) * 100

# Sinif bazli
leak_sonuclar = [s for s in sonuclar if s["gercek"] == "leak"]
no_leak_sonuclar = [s for s in sonuclar if s["gercek"] == "no_leak"]

leak_dogru = sum(1 for s in leak_sonuclar if s["dogru"])
no_leak_dogru = sum(1 for s in no_leak_sonuclar if s["dogru"])

leak_basari = (leak_dogru / len(leak_sonuclar) * 100) if leak_sonuclar else 0
no_leak_basari = (no_leak_dogru / len(no_leak_sonuclar) * 100) if no_leak_sonuclar else 0

# Confusion matrix degerleri
TP = sum(1 for s in sonuclar if s["gercek"] == "leak" and s["tahmin"] == "leak")
TN = sum(1 for s in sonuclar if s["gercek"] == "no_leak" and s["tahmin"] == "no_leak")
FP = sum(1 for s in sonuclar if s["gercek"] == "no_leak" and s["tahmin"] == "leak")
FN = sum(1 for s in sonuclar if s["gercek"] == "leak" and s["tahmin"] == "no_leak")

print("\n" + "=" * 60)
print("  SONUC RAPORU")
print("=" * 60)
print(f"  Toplam test gorseli     : {toplam}")
print(f"  Dogru tahmin            : {dogru_sayisi}")
print(f"  Yanlis tahmin           : {yanlis_sayisi}")
print(f"  GENEL BASARI            : %{genel_basari:.1f}")
print("-" * 60)
print(f"  Leak sinifi basarisi    : {leak_dogru}/{len(leak_sonuclar)} = %{leak_basari:.1f}")
print(f"  No_leak sinifi basarisi : {no_leak_dogru}/{len(no_leak_sonuclar)} = %{no_leak_basari:.1f}")
print("-" * 60)
print(f"  Confusion Matrix:")
print(f"    TP (Leak->Leak)       : {TP}")
print(f"    TN (NoLeak->NoLeak)   : {TN}")
print(f"    FP (NoLeak->Leak)     : {FP}")
print(f"    FN (Leak->NoLeak)     : {FN}")
print("=" * 60)


# --- 6. Grafikleri Ciz ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"CNN Model Test Raporu - Genel Basari: %{genel_basari:.1f}", fontsize=14, fontweight="bold")

# Grafik 1: Genel Basari Pasta Grafigi
axes[0].pie(
    [dogru_sayisi, yanlis_sayisi],
    labels=[f"Dogru ({dogru_sayisi})", f"Yanlis ({yanlis_sayisi})"],
    colors=["#4CAF50", "#F44336"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"fontsize": 11}
)
axes[0].set_title("Genel Basari Orani", fontsize=12)

# Grafik 2: Sinif Bazli Basari (Bar Chart)
siniflar = ["Leak", "No Leak"]
basarilar = [leak_basari, no_leak_basari]
renkler = ["#2196F3", "#FF9800"]

bars = axes[1].bar(siniflar, basarilar, color=renkler, width=0.5, edgecolor="black")
axes[1].set_ylim(0, 110)
axes[1].set_ylabel("Basari (%)")
axes[1].set_title("Sinif Bazli Basari", fontsize=12)
for bar, val in zip(bars, basarilar):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"%{val:.1f}", ha="center", fontsize=12, fontweight="bold")

# Grafik 3: Confusion Matrix (Heatmap)
cm = np.array([[TP, FN], [FP, TN]])
im = axes[2].imshow(cm, cmap="Blues", vmin=0)
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Leak (Tahmin)", "NoLeak (Tahmin)"])
axes[2].set_yticklabels(["Leak (Gercek)", "NoLeak (Gercek)"])
axes[2].set_title("Confusion Matrix", fontsize=12)

# Confusion matrix hucre degerlerini yaz
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig(str(BASE_DIR / "models" / "test_raporu.png"), dpi=150, bbox_inches="tight")
print(f"\n[+] Grafik kaydedildi: models/test_raporu.png")
plt.show()
