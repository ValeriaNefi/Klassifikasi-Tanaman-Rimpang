# -*- coding: utf-8 -*-
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras import layers, models
import tensorflow as tf

# Tentukan path ke dataset (sesuaikan dengan direktori lokal Anda)
dataset_dir = r'C:\Program Files\Python312\UAS Prak Pemrogjar_202251229\rimpang fix\MANDIRI_COMVIS_1'  # Ganti dengan jalur lokal dataset Anda

# Langkah 1: Membuat ImageDataGenerator untuk Augmentasi dan Preprocessing
#Membuat generator data yang memuat data gambar dari 
# direktori dan menerapkan augmentasi serta preprocessing.
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Normalisasi sesuai dengan VGG16(penyesuaian)
    rotation_range=30,                       # Rotasi gambar secara acak
    width_shift_range=0.2,                   # Geser lebar gambar secara acak
    height_shift_range=0.2,                  # Geser tinggi gambar secara acak
    shear_range=0.2,                         # Distorsi gambar secara acak
    zoom_range=0.2,                          # Zoom gambar secara acak
    horizontal_flip=True,                    # Flip gambar secara horizontal
    fill_mode='nearest',                     # Mengisi ruang kosong setelah transformasi
    validation_split=0.2                     # 20% data untuk validasi
)

# Langkah 2: Membuat Generator untuk Training dan Validation
# Data training
#train_generator: Mengambil kelas data training dari dataset 
# dengan augmentasi dan preprocessing.
train_generator = datagen.flow_from_directory(
    dataset_dir,               # Path ke folder dataset
    target_size=(224, 224),     # Ukuran gambar sesuai dengan input VGG16
    batch_size=32,             # Ukuran batch
    class_mode='categorical',  # Klasifikasi multi-kelas
    subset='training'          # Subset untuk data training
)

# Data validasi
validation_generator = datagen.flow_from_directory(
    dataset_dir,               # Path ke folder dataset
    target_size=(224, 224),     # Ukuran gambar sesuai dengan input VGG16
    batch_size=32,             # Ukuran batch
    class_mode='categorical',  # Klasifikasi multi-kelas
    subset='validation'        # Subset untuk data validasi
)

# Cek apakah dataset berhasil dimuat
print("Training Data:", train_generator.samples, "images")
print("Validation Data:", validation_generator.samples, "images")
print("Classes:", train_generator.class_indices)

# Langkah 3: Memuat Model VGG16 tanpa Layer Top (Fully Connected)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Membekukan layer-layer dasar VGG16 untuk mencegah pelatihan ulang
#agar parameter pra-latih tidak diperbarui selama pelatihan.
#base model unutk fitur extraktor
base_model.trainable = False

# Langkah 4: Membangun Model Klasifikasi
model = models.Sequential()

# Menambahkan VGG16 sebagai fitur extractor
model.add(base_model)

# Menambahkan layer Flatten untuk meratakan output
model.add(layers.Flatten())

# Menambahkan layer Dense dengan unit 512 dan fungsi aktivasi ReLU
model.add(layers.Dense(512, activation='relu'))

# Menambahkan layer Dropout untuk mengurangi overfitting
model.add(layers.Dropout(0.5))

# Menambahkan layer output dengan jumlah kelas (10 kelas) dan fungsi aktivasi softmax
model.add(layers.Dense(10, activation='softmax'))

# Langkah 5: Menyusun Model
model.compile(
    optimizer='adam',               # Optimizer Adam untuk kecepatan konvergensi
    loss='categorical_crossentropy', # Loss function untuk klasifikasi multi-kelas
    metrics=['accuracy']            # Metric yang akan digunakan untuk evaluasi
)

# Langkah 6: Melatih Model
history = model.fit(
    train_generator,                  # Data training yang sudah diproses
    steps_per_epoch=train_generator.samples // train_generator.batch_size, # Jumlah langkah per epoch
    epochs=10,                         # Jumlah epoch pelatihan
    validation_data=validation_generator,  # Data validasi
    validation_steps=validation_generator.samples // validation_generator.batch_size  # Jumlah langkah per epoch validasi
)

# Langkah 7: Evaluasi Model
train_loss, train_acc = model.evaluate(train_generator, steps=train_generator.samples // train_generator.batch_size)
val_loss, val_acc = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)

# Menampilkan hasil akurasi dan loss untuk training dan validasi
print(f'Training Accuracy: {train_acc}')
print(f'Training Loss: {train_loss}')
print(f'Validation Accuracy: {val_acc}')
print(f'Validation Loss: {val_loss}')

# Langkah 8: Menyimpan Model yang Sudah Dilatih
model.save('vgg16_rimpang_model.h5')  # Model disimpan di direktori lokal
print("Model telah disimpan di direktori lokal.")
