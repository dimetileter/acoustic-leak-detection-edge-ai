# -*- coding: utf-8 -*-
"""
CNN Modeli Rastgele Test Scripti
================================
spectrogram_pool klasorunden rastgele bir gorsel secip
leak_detection_cnn.keras modeline tahmin yaptirir.
Gercek etiketle karsilastirarak dogrulugu gosterir.
"""

import os
import sys
import random
import pathlib
import numpy as np

# Konsol encoding sorununu coz
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
import matplotlib.pyplot as plt


# --- 1. Yollari Tanimla ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "leak_detection_cnn_only_100(2.0).keras"
DATA_DIR = BASE_DIR / "dataset" / "image_datasets" / "spectrogram_pool_augmented_test_not_in_train"

# Egitim kodundaki siralama ile ayni: leak=0, no_leak=1
SINIF_ISIMLERI = ["leak", "no_leak"]


# --- 2. Modeli Yukle ---
print("=" * 55)
print("  CNN Model Test Araci")
print("=" * 55)
print(f"\n[*] Model yukleniyor: {MODEL_PATH}")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("[+] Model basariyla yuklendi!\n")


# --- 3. Rastgele Bir Klasor ve Gorsel Sec ---
# Once rastgele bir sinif sec (leak veya no_leak)
rastgele_sinif = random.choice(SINIF_ISIMLERI)
sinif_klasoru = DATA_DIR / rastgele_sinif

# O klasordeki tum PNG dosyalarini listele
dosyalar = list(sinif_klasoru.glob("*.png"))

if not dosyalar:
    print(f"[-] '{sinif_klasoru}' klasorunde PNG dosyasi bulunamadi!")
    exit(1)

# Rastgele bir dosya sec
rastgele_dosya = random.choice(dosyalar)
gercek_etiket_index = SINIF_ISIMLERI.index(rastgele_sinif)

print(f"[>] Rastgele secilen sinif : {rastgele_sinif}")
print(f"[>] Secilen dosya          : {rastgele_dosya.name}")
print(f"[>] Gercek etiket          : {rastgele_sinif} (index={gercek_etiket_index})")
print("-" * 55)


# --- 4. Gorseli Yukle ve On Isle ---
# Egitim kodundaki load_and_preprocess_image fonksiyonu ile ayni
image = tf.io.read_file(str(rastgele_dosya))
image = tf.image.decode_png(image, channels=3)
image = tf.image.resize(image, [128, 128])
image = image / 255.0  # Normalizasyon

# Model batch boyutu bekler -> (1, 128, 128, 3)
image_batch = tf.expand_dims(image, axis=0)


# --- 5. Tahmin Yap ---
tahmin_olasiligi = model.predict(image_batch, verbose=0)[0][0]

# Sigmoid ciktisi: < 0.5 => leak (0), >= 0.5 => no_leak (1)
tahmin_index = 1 if tahmin_olasiligi >= 0.5 else 0
tahmin_sinif = SINIF_ISIMLERI[tahmin_index]

# Guven yuzdesini hesapla
if tahmin_index == 1:
    guven = tahmin_olasiligi * 100
else:
    guven = (1 - tahmin_olasiligi) * 100


# --- 6. Sonuclari Yazdir ---
dogru_mu = tahmin_index == gercek_etiket_index

print(f"\n[*] Model Tahmini          : {tahmin_sinif}")
print(f"[*] Sigmoid Ciktisi        : {tahmin_olasiligi:.4f}")
print(f"[*] Guven Orani            : %{guven:.1f}")
print("-" * 55)

if dogru_mu:
    print(">>> SONUC: [DOGRU TAHMIN] <<<")
else:
    print(">>> SONUC: [YANLIS TAHMIN] <<<")

print(f"    Gercek: {rastgele_sinif}  |  Tahmin: {tahmin_sinif}")
print("=" * 55)


# --- 7. Gorseli Goster ---
plt.figure(figsize=(6, 6))
plt.imshow(image.numpy())
plt.title(
    f"Gercek: {rastgele_sinif}  |  Tahmin: {tahmin_sinif}\n"
    f"{'DOGRU' if dogru_mu else 'YANLIS'}  |  Guven: %{guven:.1f}",
    fontsize=12,
    color="green" if dogru_mu else "red",
    fontweight="bold"
)
plt.axis("off")
plt.tight_layout()
plt.show()
