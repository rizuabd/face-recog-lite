import os
# Fix UI Error di Raspberry Pi
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# --- KONFIGURASI ---
RTSP_URL = "rtsp://admin:idslci123@192.168.0.3:554/h264Preview_01_sub"
MODEL_PATH = "model/facenet_128_float32.tflite"
DB_FILE = "wajah_database.npz"

# --- FUNGSI RECOGNITION (SMART QUANTIZATION) ---
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
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=True)
        db = dict(data['db'].item())
        print(f"[INFO] Database dimuat. Total wajah: {len(db)}")
    else:
        db = {}
        print("[INFO] Database baru dibuat.")

    nama_user = input("Masukkan Nama untuk wajah ini: ")
    print("Silakan berdiri di depan CCTV. Tekan 's' untuk save, 'q' untuk batal.")

    cap = cv2.VideoCapture(RTSP_URL)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame_small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        target_face = None

        for (x, y, w, h) in faces:
            cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)
            target_face = frame_small[y:y+h, x:x+w]

        cv2.imshow("Registrasi Wajah - Tekan 's' untuk Save", frame_small)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and target_face is not None:
            vector = get_embedding(interpreter, input_details, output_details, target_face)
            db[nama_user] = vector
            np.savez(DB_FILE, db=db)
            print(f"[SUKSES] Wajah '{nama_user}' berhasil disimpan!")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()