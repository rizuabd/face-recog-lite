import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# --- KONFIGURASI ---
FACES_DIR = "faces"
MODEL_PATH = "model/facenet_128_float32.tflite"
DB_FILE = "wajah_database.npz"

# --- FUNGSI UTAMA YANG SUDAH DI-UPGRADE ---
def get_embedding(interpreter, input_details, output_details, img):
    # 1. Ambil Ukuran
    input_shape = input_details[0]['shape']
    h, w = input_shape[1], input_shape[2] # 160x160
    
    # 2. Preprocess
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    # Normalisasi MobileNetV2 (-1 s/d 1)
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    
    # 3. Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # 4. Output
    # Karena L2 Norm udah ada di dalam model (layer Lambda terakhir),
    # kita tinggal ambil mentahannya aja.
    return interpreter.get_tensor(output_details[0]['index'])[0]
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model {MODEL_PATH} tidak ditemukan!")
        return

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    new_db = {}
    
    if not os.path.exists(FACES_DIR):
        print(f"[ERROR] Folder '{FACES_DIR}' tidak ditemukan.")
        return

    people = [d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))]
    print(f"[INFO] Ditemukan {len(people)} folder nama: {people}")

    for person_name in people:
        person_folder = os.path.join(FACES_DIR, person_name)
        image_paths = [os.path.join(person_folder, f) for f in os.listdir(person_folder) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        embeddings = []
        print(f"Processing: {person_name} ({len(image_paths)} gambar)...")

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                face_roi = img[y:y+h, x:x+w]
                emb = get_embedding(interpreter, input_details, output_details, face_roi)
                embeddings.append(emb)
            else:
                print(f"  [WARN] Wajah tidak terdeteksi di {img_path}, skip.")

        if len(embeddings) > 0:
            avg_embedding = np.mean(embeddings, axis=0)
            new_db[person_name] = avg_embedding
            print(f"  [OK] {person_name} terdaftar.")
        else:
            print(f"  [FAIL] Gagal mendaftarkan {person_name}.")

    if len(new_db) > 0:
        np.savez(DB_FILE, db=new_db)
        print(f"\n[SUKSES] Database disimpan ke '{DB_FILE}'. Total: {len(new_db)} orang.")
    else:
        print("\n[INFO] Tidak ada data baru yang disimpan.")

if __name__ == "__main__":
    main()