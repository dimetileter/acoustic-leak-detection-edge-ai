#%%
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import random
import os
import numpy as np

from tensorflow.python.keras.callbacks import EarlyStopping

#%%
# 1. Pathlib kullanarak yolu tanımla
_BASE = pathlib.Path(__file__).resolve().parent.parent
data_dir = _BASE / "dataset" / "image_datasets" / "spectrogram_pool_augmented_from_only_100"

# 2. Dosya yollarını ve etiketleri topla
all_image_paths = list(data_dir.glob('*/*.png'))
all_image_paths = [str(path) for path in all_image_paths]

# Sınıf isimleri: leak=0, no_leak=1
label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

# Veriyi karıştır
c = list(zip(all_image_paths, all_image_labels))
random.seed(42)
random.shuffle(c)
all_image_paths, all_image_labels = zip(*c)
all_image_paths = list(all_image_paths)
all_image_labels = list(all_image_labels)

# 3. Görsel yükleme (MobileNetV2 için 128x128, [0,1] aralığı)
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image /= 255.0
    return image, label

path_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
image_label_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# 4. Eğitim / Doğrulama ayrımı
DATASET_SIZE = len(all_image_paths)
train_size = int(0.8 * DATASET_SIZE)

train_ds = image_label_ds.take(train_size).batch(16).prefetch(tf.data.AUTOTUNE)
val_ds = image_label_ds.skip(train_size).batch(16).prefetch(tf.data.AUTOTUNE)

print(f"Toplam {DATASET_SIZE} resim yüklendi. Sınıflar: {label_names}")

# 5. Sınıf Ağırlıkları
labels_array = np.array(all_image_labels)
n_leak = np.sum(labels_array == 0)
n_no_leak = np.sum(labels_array == 1)
total = len(labels_array)

class_weight = {
    0: total / (2.0 * n_leak),
    1: total / (2.0 * n_no_leak)
}
print(f"Sınıf Ağırlıkları: leak={class_weight[0]:.2f}, no_leak={class_weight[1]:.2f}")

from tensorflow.keras import layers, models

#%%
# ================================================================
# TRANSFER LEARNING — MobileNetV2
# ================================================================
# ImageNet üzerinde milyonlarca görsel ile önceden eğitilmiş model.
# Temel görsel öznitelikleri zaten bilir (kenarlar, dokular, paternler).
# Biz sadece son katmanları "leak vs no_leak" için eğitiyoruz.

# Önceden eğitilmiş MobileNetV2 modelini yükle
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,          # Son sınıflandırma katmanını dahil etme
    weights='imagenet'          # ImageNet ağırlıklarını kullan
)

# Temel modelin ağırlıklarını dondur (eğitim sırasında değişmesin)
base_model.trainable = False

print(f"\nMobileNetV2 yüklendi: {len(base_model.layers)} katman (hepsi donduruldu)")

# Yeni sınıflandırma katmanları ekle
model = models.Sequential([
    base_model,

    # Global Average Pooling: Her öznitelik haritası tek bir değere indirgenir
    layers.GlobalAveragePooling2D(),

    # Sınıflandırma
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Derleme
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Callbacks ---
early_stopper = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ===========================
# Sadece üst katmanları eğit (base model dondurulmuş)
# Fine-tuning küçük veri setlerinde overfitting'e neden olduğu için kaldırıldı
# ===========================
print("\n" + "="*50)
print("Üst katmanları eğitme (MobileNetV2 dondurulmuş)")
print("="*50)

epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weight,
    callbacks=[early_stopper],
    workers=4
)

# Modeli kaydet
current_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_dir, "leak_detection_transfer.keras")
model.save(model_save_path)
print(f"\nModel kaydedildi: {model_save_path}\n")

# Sonuçları Grafikleştirme
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.legend()
plt.title('Doğruluk (Accuracy)')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp (Loss)')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()
