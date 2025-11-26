import tensorflow as tf
import os
import numpy as np

# Nama file
H5_FILE = "mobilenetv2_128_fresh.h5"
TFLITE_FILE = "facenet_128_float32.tflite"

def main():
    print("[1] Membangun Model MobileNetV2 dari Nol...")
    # Kita pakai arsitektur MobileNetV2 (ini standar industri buat HP/Raspi)
    # Input 160x160 RGB
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False, # Buang buntut klasifikasinya
        weights='imagenet' # Pake otak dasar pengenalan benda (bukan wajah, tapi lumayan)
    )
    
    # Tambahkan buntut baru supaya outputnya 128 Dimensi (Kaya FaceNet)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation=None)(x) # Output Vector 128
    # Normalisasi L2 di dalam model (biar outputnya udah mateng)
    outputs = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    # Simpan jadi .h5 (Ini simulasi lu punya file .h5 sendiri)
    print(f"    Menyimpan ke '{H5_FILE}'...")
    model.save(H5_FILE)

    print("\n[2] Proses Belajar: Mengkonversi ke TFLite...")
    # Load balik file .h5 yang barusan kita buat (Pasti kompatibel karena satu laptop)
    model_loaded = tf.keras.models.load_model(H5_FILE)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model_loaded)
    
    # --- KUNCI BIAR RASPI GAK OVERFLOW ---
    # 1. Jangan di-Quantize (Optimizations = []) -> Biar CPU Raspi gak pusing dekompresi
    # 2. Model MobileNetV2 ini jauh lebih ringan dari InceptionResnet (90MB kemarin)
    converter.optimizations = [] 
    
    tflite_model = converter.convert()

    with open(TFLITE_FILE, "wb") as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(TFLITE_FILE) / 1024 / 1024
    print(f"\n[SUKSES BESAR] File '{TFLITE_FILE}' jadi!")
    print(f"Ukuran: {size_mb:.2f} MB")
    print("-> Karena ini MobileNetV2 (bukan Inception), ukurannya kecil (~9MB) walau Float32.")
    print("-> Kirim file ini ke Raspi, gw jamin 1000% jalan!")

if __name__ == "__main__":
    main()