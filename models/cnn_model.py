#%%
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import random  # Karıştırma işlemi için eklendi


import os

from tensorflow.python.keras.callbacks import EarlyStopping

#%%
# 1. Pathlib kullanarak yolu tanımla (Türkçe karakter sorununu aşar)
# __file__ ile mutlak yol: scriptin hangi dizinden çalıştırıldığına bağlı kalmaz
_BASE = pathlib.Path(__file__).resolve().parent.parent  # proje kök dizini
data_dir = _BASE / "dataset" / "image_datasets" / "spectrogram_pool_augmented_from_only_100"

# 2. Dosya yollarını ve etiketleri manuel olarak topla
# görselindeki yapıya sadık kalıyoruz
all_image_paths = list(data_dir.glob('*/*.png'))
all_image_paths = [str(path) for path in all_image_paths]

# Klasör isimlerine göre etiketleri belirle (0: leak, 1: no_leak)
label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

# --- KRİTİK ÇÖZÜM: VERİYİ KARIŞTIR (SHUFFLE) ---
# Yolları ve etiketleri birbirine bağlayıp karıştırıyoruz, sonra tekrar ayırıyoruz.
c = list(zip(all_image_paths, all_image_labels))
random.seed(42) # Her çalıştırmada aynı rastgeleliği elde etmek için (tutarlılık sağlar)
random.shuffle(c)
all_image_paths, all_image_labels = zip(*c)

all_image_paths = list(all_image_paths)
all_image_labels = list(all_image_labels)
# -----------------------------------------------

# 3. TensorFlow Veri Setini Oluştur
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image /= 255.0  # Normalizasyon: İş akış planı madde 2.1
    return image, label

path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
image_label_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# 4. Eğitim ve Doğrulama Setlerine Ayır
DATASET_SIZE = len(all_image_paths)
train_size = int(0.8 * DATASET_SIZE)
val_size = DATASET_SIZE - train_size

train_ds = image_label_ds.take(train_size).batch(8).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = image_label_ds.skip(train_size).batch(8).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Toplam {DATASET_SIZE} resim yüklendi. Sınıflar: {label_names}")

# --- Sınıf Ağırlıkları (Class Weight) ---
# Modelin az temsil edilen sınıfı daha iyi öğrenmesi için
import numpy as np
labels_array = np.array(all_image_labels)
n_leak = np.sum(labels_array == 0)
n_no_leak = np.sum(labels_array == 1)
total = len(labels_array)

class_weight = {
    0: total / (2.0 * n_leak),     # leak ağırlığı
    1: total / (2.0 * n_no_leak)   # no_leak ağırlığı
}
print(f"Sınıf Ağırlıkları: leak={class_weight[0]:.2f}, no_leak={class_weight[1]:.2f}")

from tensorflow.keras import layers, models

#%%
# 1. CNN Model Mimarisi (Basit mimari + Dropout + düşük LR)
model = models.Sequential([
    # İlk Evrişim Katmanı
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # İkinci Evrişim Katmanı
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Üçüncü Evrişim Katmanı
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Sınıflandırma
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 2. Derleme — düşük öğrenme hızı
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 3. Eğitimi Başlat
epochs = 12
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weight,
    callbacks=[early_stopper],
    workers=4
)

current_dir = os.path.dirname(os.path.abspath(__file__))
# Modelin ismini ve uzantısını belirliyoruz
model_save_path = os.path.join(current_dir, "leak_detection_cnn_only_100(2.0).keras")

# Modeli diske yazdırıyoruz
model.save(model_save_path)
print(f"\nModel başarıyla şuraya kaydedildi: {model_save_path}\n")

# 4. Sonuçları Grafikleştirme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.legend()
plt.title('Doğruluk (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp (Loss)')
plt.show()