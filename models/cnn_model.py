#%%
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

#%%
# 1. Pathlib kullanarak yolu tanımla (Türkçe karakter sorununu aşar)
data_dir = pathlib.Path(r"../dataset/spectrogram_pool")

# 2. Dosya yollarını ve etiketleri manuel olarak topla
# görselindeki yapıya sadık kalıyoruz
all_image_paths = list(data_dir.glob('*/*.png'))
all_image_paths = [str(path) for path in all_image_paths]

# Klasör isimlerine göre etiketleri belirle (0: leak, 1: no_leak)
label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

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

train_ds = image_label_ds.take(train_size).batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = image_label_ds.skip(train_size).batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)

print(f"Toplam {DATASET_SIZE} resim yüklendi. Sınıflar: {label_names}")

from tensorflow.keras import layers, models

#%%
# 1. CNN Model Mimarisi Oluşturma (İş Akış Planı Madde 2.2)
model = models.Sequential([
    # Giriş katmanı: 128x128 boyutunda, 3 kanal (RGB)
    # Normalizasyon zaten manuel yapıldı (0-1 arası) [cite: 32]

    # İlk Evrişim Katmanı (Öznitelik Çıkarım) [cite: 34]
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    # İkinci Evrişim Katmanı (Desen Tespiti) [cite: 35]
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Üçüncü Evrişim Katmanı
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Vektörleştirme (Madde 2.2.36) [cite: 36]
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6),  # Ezberlemeyi (overfitting) önlemek için
    layers.Dense(1, activation='sigmoid')  # 2 sınıf: leak veya no_leak [cite: 6]
])

# 2. Derleme
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

#

# 3. Eğitimi Başlat
epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

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