# 📌 Pothole Image Detection using CNN and Transfer Learning 

## 1. Project Overview
Proyek ini bertujuan untuk mengembangkan model deteksi lubang jalan (*pothole detection*) berbasis citra menggunakan Deep Learning, khususnya **Convolutional Neural Networks (CNN)** dan **transfer learning** dari model pretrained seperti **VGG16** dan **MobileNetV2**.  
Proyek ini diharapkan dapat membantu proses identifikasi lubang jalan secara otomatis dan efisien.

---

## 2. Dataset
- **Sumber**: Dataset **kaggle** berisi gambar jalan normal dan jalan berlubang 
- **Jumlah Kelas**: 
  - `0` → Jalan Normal
  - `1` → Jalan Berlubang (Pothole)
- **Preprocessing**:
  - Resize gambar menjadi 224x224 piksel.
  - Normalisasi pixel (`rescale=1/255`).
  - Augmentasi data: rotasi, zoom, horizontal flip, dan shift.

---

## 3. Modeling Approach

### a. CNN from Scratch
- Model CNN sederhana digunakan sebagai baseline awal.
- Observasi: akurasi bagus, namun cenderung overfitting.

### b. Transfer Learning
- **Model yang digunakan**:
  - VGG16
  - MobileNetV2
- **Strategi**:
  - Membekukan (*freeze*) bobot pretrained.
  - Menambahkan layer baru:
    - GlobalAveragePooling2D
    - Dense 128 neurons + ReLU
    - Dropout (rate 0.3)
    - Dense output layer (Sigmoid)

---

## 4. Training Details
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Validation Split**: 25% dari training set
- **Early Stopping**: tidak digunakan (opsional untuk pengembangan lanjut)

---

## 5. Evaluation Metrics
- **Accuracy**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **Training History Plot** (Loss & Accuracy)

---

## 6. Results
| Model | Test Accuracy | Observations |
|:------|:--------------|:-------------|
| VGG16 | ~94-95%        | Akurasi tinggi namun inference time lebih lambat, model berat |
| MobileNetV2 | ~95-96%  | Akurasi tinggi dengan model yang jauh lebih ringan dan cepat |

> **Selected Model**: **VGG16** untuk deployment karena performa lebih optimal di resource terbatas.

---

## 7. Deployment
- **Framework**: Streamlit
- **Platform**: Hugging Face Spaces
- **Features**:
  - Dua halaman: EDA dan Pothole Prediction.
  - EDA menampilkan:
    - Sample images dari kedua kelas.
    - Statistik dasar dataset.
    - Distribusi kelas.
  - Halaman Prediksi:
    - Upload gambar jalan.
    - Pilihan untuk memilih gambar contoh.
    - Hasil prediksi apakah ada lubang atau tidak.

---

## 8. Limitations and Future Work
- **Limitasi**:
  - Model kadang salah deteksi pada gambar ambigu (bayangan, air di jalan).
  - Belum dilakukan optimasi model untuk edge device.
- **Future Improvements**:
  - Fine-tuning model pretrained.
  - Training dengan dataset yang lebih besar dan bervariasi.
  - Implementasi teknik quantization/pruning untuk optimasi model.
  - Membangun pipeline monitoring untuk menangani data drift.

---

## 9. Directory Structure
```
deployment/ - Folder berisi file deployment aplikasi streamlit di huggingface spaces
├── sample_images/
├── data/
├── app.py
├── eda.py
├── prediction.py
├── requirements.txt
├── deeplearning.png
├── best_model.h5.txt
notebook_project_muhammad_iqbal.ipynb - Notebook ini berisi proses pengerjaan projek kali ini
inference_muhammad_iqbal.ipynb - Notebook ini berisi proses inference model yang sudah dilatih
README.md - Dokumentasi projek ini dalam format Markdown
```

---

## 10. Conclusion
Penggunaan transfer learning dengan pretrained model seperti VGG16 dan MobileNetV2 memberikan hasil yang sangat baik dalam mendeteksi lubang jalan.  
Untuk penggunaan produksi, VGG16 direkomendasikan karena keseimbangan antara akurasi dan efisiensi komputasi.

## Reference
Dataset - [Pothole image detection](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset/)
Deployment - [HuggingFace](https://huggingface.co/spaces/mbale014/Pothole-Detection-by-mbale014)
VGG16 - [Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman (2015)](https://arxiv.org/abs/1409.1556)
MobileNetV2 - [Inverted Residuals and Linear Bottlenecks
M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L. Chen (2019)](https://arxiv.org/abs/1801.04381)
Keras Applications Documentation - [Keras Applications](https://keras.io/api/applications/)
Streamlit Documentation - [Streamlit](https://docs.streamlit.io/)
Hugging Face Spaces Documentation - [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)


---