import os
# Fix UI Error
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import threading
import queue
import time

# --- CONFIG ---
# Pilih salah satu URL di bawah ini (Uncomment yang mau dipake)
# RTSP_URL = "rtsp://admin:idslci123@192.168.0.3:554/h264Preview_01_sub"
RTSP_URL = 0 # Pakai ini kalau mau test webcam laptop/USB

MODEL_PATH = "model/facenet_128_float32.tflite"
DB_FILE = "wajah_database.npz"
THRESHOLD = 0.8 

class VideoStream:
    def __init__(self, path):
        self.stream = cv2.VideoCapture(path)
        self.q = queue.Queue()
        self.stopped = False
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
    def update(self):
        while True:
            if self.stopped: return
            if not self.stream.isOpened(): continue
            (grabbed, frame) = self.stream.read()
            if not grabbed: continue
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)
    def read(self):
        try: return self.q.get(timeout=1)
        except queue.Empty: return None
    def stop(self):
        self.stopped = True
        self.stream.release()

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
def match_face(embedding, db):
    min_dist = 100
    identity = "Unknown"
    
    # print(f"--- Debug ---") 
    for name, db_emb in db.items():
        dist = np.linalg.norm(embedding - db_emb)
        # print(f"Jarak ke '{name}': {dist:.4f}") 
        
        if dist < min_dist:
            min_dist = dist
            if dist < THRESHOLD:
                identity = name
    
    return identity, min_dist

def main():
    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=True)
        db = dict(data['db'].item())
        print(f"[INFO] Database dimuat: {list(db.keys())}")
    else:
        print("[WARNING] Database wajah kosong! Jalankan register_face.py dulu.")
        db = {}

    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"Error load model: {e}")
        return

    cam = VideoStream(RTSP_URL)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    time.sleep(1.0)

    print("[INFO] Recognition Start...")
    
    while True:
        frame = cam.read()
        if frame is None: continue

        frame_small = cv2.resize(frame, (720, 1080))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_roi = frame_small[y:y+h, x:x+w]
            
            try:
                emb = get_embedding(interpreter, input_details, output_details, face_roi)
                name, dist = match_face(emb, db)
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({dist:.2f})"
                
                cv2.rectangle(frame_small, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_small, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(e)

        cv2.imshow("Face Recognition", frame_small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
