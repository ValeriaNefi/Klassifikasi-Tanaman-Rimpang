# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Langkah 1: Path ke Model dan Dataset
model_path = r'C:\Program Files\Python312\UAS Prak Pemrogjar_202251229\rimpang fix\vgg16_rimpang_model.h5'  # Sesuaikan path model
dataset_dir = r'C:\Program Files\Python312\UAS Prak Pemrogjar_202251229\rimpang fix\Test'  # Sesuaikan path dataset

# Langkah 2: Muat Model yang Telah Dilatih
model = load_model(model_path)

# Langkah 3: Membuat Data Generator untuk Data Testing
#normalisasi
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Langkah 4: Muat Data Testing
#
test_generator = test_datagen.flow_from_directory(
    dataset_dir,                 # Path ke folder dataset
    target_size=(224, 224),      # Ukuran gambar sesuai input model
    batch_size=32,               # Ukuran batch
    class_mode='categorical',    # Klasifikasi multi-kelas
    #Data tidak diacak, sehingga prediksi dapat dibandingkan dengan label asli.
    shuffle=False                # Jangan shuffle untuk evaluasi
)

# Langkah 5: Prediksi pada Data Testing
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  # Prediksi kelas
true_classes = test_generator.classes               # Kelas sebenarnya
class_labels = list(test_generator.class_indices.keys())  # Label kelas

# Langkah 6: Evaluasi Model
# Proporsi prediksi yang benar.
accuracy = accuracy_score(true_classes, predicted_classes)
# Proporsi prediksi benar dari semua prediksi untuk setiap kelas.
precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=1)
#Proporsi prediksi benar dari semua sampel sebenarnya untuk setiap kelas.
recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=1)
# Harmonic mean antara precision dan recall
f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=1)

# Menampilkan Hasil Evaluasi
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Menampilkan Classification Report
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=1))

# Langkah 7: Visualisasi Gambar dengan Prediksi
def visualize_predictions(generator, predictions, true_classes, class_labels, num_images=16):
    """
    Menampilkan gambar asli dari data testing dengan prediksi.
    Hijau untuk prediksi benar, merah untuk prediksi salah.
    """
    fig, axes = plt.subplots(int(np.ceil(num_images / 4)), 4, figsize=(15, 15))  # Grid dinamis sesuai jumlah gambar
    axes = axes.ravel()

    # Batasi jumlah gambar sesuai input
    max_images = min(num_images, len(true_classes))
    for i in range(max_images):
        # Muat gambar asli (bukan hasil augmentasi)
        image_path = generator.filepaths[i]
        image = plt.imread(image_path)

        # Prediksi dan label asli
        true_label = true_classes[i]
        pred_label = np.argmax(predictions[i])

        # Tampilkan gambar
        axes[i].imshow(image)
        axes[i].axis('off')

        # Warna teks
        color = "green" if true_label == pred_label else "red"
        title = f"True: {class_labels[true_label]}\nPred: {class_labels[pred_label]}"
        axes[i].set_title(title, color=color)

    # Hapus grid kosong jika gambar lebih sedikit dari grid
    for j in range(max_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Tentukan jumlah gambar yang ingin ditampilkan
num_images_to_display = 16
visualize_predictions(test_generator, predictions, true_classes, class_labels, num_images=num_images_to_display)

def visualize_predictions_by_class(generator, predictions, true_classes, class_labels, target_class=None, num_images=16):
    """
    Menampilkan gambar asli dari data testing berdasarkan kelas tertentu.
    Hijau untuk prediksi benar, merah untuk prediksi salah.
    """
    fig, axes = plt.subplots(int(np.ceil(num_images / 4)), 4, figsize=(15, 15))  # Grid dinamis sesuai jumlah gambar
    axes = axes.ravel()

    # Filter gambar berdasarkan kelas target
    selected_indices = []
    for i in range(len(true_classes)):
        true_label = true_classes[i]
        pred_label = np.argmax(predictions[i])
        if target_class is None or class_labels[true_label] == target_class or class_labels[pred_label] == target_class:
            selected_indices.append(i)

    # Batasi jumlah gambar sesuai input
    max_images = min(num_images, len(selected_indices))
    for j, idx in enumerate(selected_indices[:max_images]):
        # Path gambar asli
        image_path = generator.filepaths[idx]
        image = plt.imread(image_path)  # Membaca gambar asli

        # Prediksi dan label asli
        true_label = true_classes[idx]
        pred_label = np.argmax(predictions[idx])

        # Tampilkan gambar
        axes[j].imshow(image)
        axes[j].axis('off')

        # Warna teks
        color = "green" if true_label == pred_label else "red"
        title = f"True: {class_labels[true_label]}\nPred: {class_labels[pred_label]}"
        axes[j].set_title(title, color=color)

    # Hapus grid kosong jika gambar lebih sedikit dari grid
    for k in range(max_images, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()

# Contoh Penggunaan
# Menampilkan hasil prediksi untuk kelas tertentu (misalnya 'kencur')
target_class = "jahe"  # Ganti dengan nama kelas lain, misalnya 'kunyit'
num_images_to_display = 16  # Jumlah gambar yang ingin ditampilkan
visualize_predictions_by_class(test_generator, predictions, true_classes, class_labels, target_class, num_images_to_display)
