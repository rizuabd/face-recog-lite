## IoT Face Recognition System (MobileNetV2 + TFLite)

Project ini adalah implementasi sistem pengenalan wajah (Face Recognition) yang ringan dan efisien, dirancang khusus untuk berjalan di perangkat Edge seperti Raspberry Pi 4/5.

Sistem ini menggunakan arsitektur MobileNetV2 (128-dimensi) yang dibangun dan dikonversi secara khusus untuk menghindari masalah kompatibilitas memori (Overflow) pada TFLite Runtime di Raspberry Pi.

### Struktur Direktori

Project ini dibagi menjadi dua lingkungan kerja: Laptop (Build) dan Raspberry Pi (Run).
```
face-recog/
├── build/                      # [LAPTOP] Area kerja untuk membuat model AI
│   ├── 0-build_and_convert.py  # Script untuk build arsitektur & convert ke TFLite
│   └── requirements.txt        # Library yang dibutuhkan di Laptop (TensorFlow, dll)
│
├── model/                      # [LAPTOP] Folder penyimpanan sementara model mentah
│   └── mobilenetv2_128_fresh.h5
│
├── rpi-program/                # [RASPBERRY PI] Program utama yang dijalankan di alat
│   ├── faces/                  # Folder dataset wajah (Organisir per nama folder)
│   │   └── pak bahlil/         # Contoh: Folder berisi foto Pak Bahlil
│   ├── model/                  # Tempat menaruh file .tflite hasil convert
│   │   └── facenet_128_float32.tflite
│   ├── folder_regis.py         # Script pendaftaran wajah dari folder foto
│   ├── register_face.py        # Script pendaftaran wajah via kamera langsung
│   ├── main.py                 # Program utama CCTV / Face Recognition
│   └── requirements.txt        # Library ringan untuk Raspi (tflite-runtime, opencv)
│
└── readme.md                   # Dokumentasi ini
```

### Bagian 1: Development (Dilakukan di Laptop/Mac)

Tahap ini bertujuan untuk membangun "Otak" AI dari nol, menyimpannya ke format Keras (.h5), lalu mengonversinya menjadi TFLite (.tflite) yang kompatibel dengan Raspberry Pi.

#### 1. Persiapan Environment

Pastikan menggunakan Python 3.11 (Direkomendasikan).

```
cd build
pip install -r requirements.txt
```

(Isi requirements.txt: tensorflow, numpy, opencv-python)

#### 2. Build & Convert Model

Jalankan script untuk membuat model:
```
python 0-build_and_convert.py
```
Apa yang script ini lakukan?
- Mendownload base model MobileNetV2 (ImageNet weights).
- Memodifikasi output layer menjadi 128 dimensi + L2 Normalization.
- Menyimpan model ke ../model/mobilenetv2_128_fresh.h5.
- Mengonversi model tersebut menjadi TFLite Float32 (Uncompressed).

Note: Kita menggunakan format Float32 tanpa kompresi untuk menjamin kompatibilitas 100% dan menghindari error CONV_2D Overflow di Raspberry Pi.

Hasil akhir: facenet_128_float32.tflite.

#### 3. Transfer File

Pindahkan folder rpi-program (beserta isinya) dan file .tflite yang baru dibuat ke Raspberry Pi.

###  Bagian 2: Deployment (Dilakukan di Raspberry Pi)

Tahap ini adalah menjalankan sistem pengenalan wajah secara real-time.

#### 1. Setup Raspberry Pi

Masuk ke folder project di Raspi dan install dependensi ringan.
```
cd ~/face-recog/rpi-program
pip install -r requirements.txt
```

(Isi requirements.txt: tflite-runtime, opencv-python, numpy)

#### 2. Persiapan Database Wajah

Ada dua cara untuk mendaftarkan wajah:

#### Cara A: Lewat Folder (Disarankan)

Siapkan foto orang yang ingin dikenali.
Buat folder dengan nama orang tersebut di dalam direktori faces/.
```
Contoh: faces/pak bahlil/foto1.jpg
```

Jalankan script registrasi:
```
python folder_regis.py
```

Script ini akan membaca foto, mendeteksi wajah, meng-extract fitur (embedding), dan menyimpannya ke wajah_database.npz.

#### Cara B: Lewat Kamera Langsung

Jalankan script:
```
python register_face.py
```

Ikuti instruksi di layar untuk mengambil sampel wajah secara live.

#### 3. Menjalankan Face Recognition

Setelah database wajah_database.npz terbentuk, jalankan program utama:
```
python main.py
```
Fitur Utama main.py:

- Terhubung ke RTSP Stream CCTV (Anti-lag dengan Threading).

- Deteksi wajah menggunakan Haar Cascade.

- Pengenalan identitas menggunakan model TFLite Custom.

- Menampilkan nama dan jarak kemiripan (distance).

- Distance < 0.8 = Wajah Dikenali (Match).

- Distance > 0.8 = Unknown.

#### Catatan Teknis (Troubleshooting)

Kenapa menggunakan MobileNetV2 Float32?

Dalam pengembangan ini, ditemukan bahwa model Inception ResNet (512-dim) terlalu berat untuk TFLite Runtime di Raspberry Pi, menyebabkan error BytesRequired number of elements overflowed.

Solusinya adalah:
```
- Mengganti arsitektur ke MobileNetV2 (128-dim) yang jauh lebih ringan.

- Menghindari Quantization (Int8) dan tetap menggunakan Float32 agar tidak terjadi konflik versi Opcode dengan library Raspberry Pi.
```
#### Cara Ganti Model (Upgrade)

Jika ingin menggunakan model yang lebih pintar (sudah dilatih jutaan wajah), cukup ganti file facenet_128_float32.tflite di folder rpi-program/model/ dengan file FaceNet.tflite (dari repo shubham0204), lalu jalankan python folder_regis.py ulang untuk memperbarui database.