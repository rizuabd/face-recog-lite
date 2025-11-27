import os
# Fix UI Error on Linux/Jetson
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import threading
import queue
import time
from flask import Flask, render_template_string, jsonify
from pyngrok import ngrok, conf

# --- COMPATIBILITY FIX ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- CONFIG ---
RTSP_URL = "rtsp://192.168.43.1:1935/"
# RTSP_URL = 0  # Ganti ke 0 untuk Webcam, atau URL RTSP
MODEL_PATH = "model/facenet_128_float32.tflite"
DB_FILE = "wajah_database.npz"
THRESHOLD = 0.8 

# --- NGROK CONFIG ---
# ⚠️ PASTE YOUR TOKEN HERE ⚠️
NGROK_AUTH_TOKEN = "315x5eHq78d0vKKHGAyfeclIGUX_2Arjp6SaCBpUKqekJ2Jk7" 
PORT = 5000

# --- GLOBAL VARS ---
app = Flask(__name__)
current_names = [] # Menyimpan data nama untuk dikirim ke web
lock = threading.Lock()

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

# --- RECOGNITION LOGIC ---
def get_embedding(interpreter, input_details, output_details, img):
    input_shape = input_details[0]['shape']
    h, w = input_shape[1], input_shape[2] 
    
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def match_face(embedding, db):
    min_dist = 100
    identity = "Unknown"
    for name, db_emb in db.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist:
            min_dist = dist
            if dist < THRESHOLD:
                identity = name
    return identity, min_dist

# --- MAIN LOOP (Optimized for No Display) ---
def detect_motion():
    global current_names, lock
    
    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=True)
        db = dict(data['db'].item())
        print(f"[INFO] Database loaded: {list(db.keys())}")
    else:
        print("[WARNING] Database empty!")
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

    print("[INFO] Recognition System Started (Headless Mode)...")
    
    while True:
        frame = cam.read()
        if frame is None: continue

        # Resize tetap dilakukan agar deteksi wajah lebih cepat (CPU lebih ringan)
        # Tapi kita tidak perlu menyimpannya untuk ditampilkan
        frame_small = cv2.resize(frame, (720, 1080))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        detected_in_this_frame = []

        for (x, y, w, h) in faces:
            face_roi = frame_small[y:y+h, x:x+w]
            try:
                emb = get_embedding(interpreter, input_details, output_details, face_roi)
                name, dist = match_face(emb, db)
                
                # Simpan hasil deteksi ke list sementara
                # Format: "Nama" (tanpa jarak biar bersih di UI, atau tambah dist jika perlu)
                if name != "Unknown":
                    detected_in_this_frame.append(f"{name} ({int((1-dist)*100)}%)")
                else:
                    detected_in_this_frame.append("Unknown")
                    
                # NOTE: Kita HAPUS bagian cv2.rectangle dan cv2.putText
                # karena tidak ada video yang ditampilkan. Ini menghemat resource.
            except Exception:
                pass

        # Update variable global agar bisa dibaca Flask
        with lock:
            current_names = list(set(detected_in_this_frame))

        # Opsional: Beri jeda sedikit agar CPU tidak 100% jika tidak perlu real-time ekstrem
        # time.sleep(0.01) 

    cam.stop()

# --- FLASK ROUTES ---
@app.route("/get_names")
def get_names():
    global current_names, lock
    with lock:
        return jsonify(current_names)

@app.route("/")
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Security Log</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { 
                        background: #121212; 
                        color: #e0e0e0; 
                        font-family: 'Segoe UI', sans-serif; 
                        display: flex; 
                        justify-content: center; 
                        padding-top: 50px;
                    }
                    .card {
                        background: #1e1e1e;
                        width: 90%;
                        max-width: 500px;
                        padding: 30px;
                        border-radius: 15px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                        text-align: center;
                    }
                    h1 { margin-bottom: 20px; font-size: 24px; color: #fff; }
                    .status-box {
                        margin-top: 20px;
                        min-height: 100px;
                    }
                    ul { list-style: none; padding: 0; }
                    li {
                        background: #2c2c2c;
                        margin: 8px 0;
                        padding: 15px;
                        border-radius: 8px;
                        font-size: 18px;
                        font-weight: bold;
                        border-left: 5px solid #007bff;
                        animation: fadeIn 0.5s;
                    }
                    li.unknown { border-left-color: #dc3545; color: #ff6b6b; }
                    li.empty { background: transparent; color: #777; font-style: italic; border: none; }
                    
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                </style>
            </head>
            <body>
                <div class="card">
                    <h1>Live Face Detection</h1>
                    <p style="color:#aaa; font-size:0.9em;">Real-time feed from Jetson Nano</p>
                    <hr style="border-color:#333;">
                    
                    <div class="status-box">
                        <ul id="nameList">
                            <li class="empty">Scanning area...</li>
                        </ul>
                    </div>
                </div>

                <script>
                    setInterval(async function() {
                        try {
                            let response = await fetch('/get_names');
                            let names = await response.json();
                            
                            let listHtml = "";
                            if (names.length === 0) {
                                listHtml = "<li class='empty'>No faces detected</li>";
                            } else {
                                names.forEach(name => {
                                    let cssClass = name.includes("Unknown") ? "unknown" : "";
                                    listHtml += `<li class="${cssClass}">${name}</li>`;
                                });
                            }
                            document.getElementById("nameList").innerHTML = listHtml;
                        } catch (e) { console.log(e); }
                    }, 1000); 
                </script>
            </body>
        </html>
    ''')

# --- MAIN ---
if __name__ == "__main__":
    # Start Thread
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()

    # Start Ngrok
    MY_TOKEN = "315x5eHq78d0vKKHGAyfeclIGUX_2Arjp6SaCBpUKqekJ2Jk7" 
    conf.get_default().auth_token = MY_TOKEN
    
    try:
        public_url = ngrok.connect(PORT).public_url
        print(f"\n[INFO] 🌍 Dashboard Link: {public_url} \n")
    except Exception as e:
        print(f"[ERROR] Ngrok: {e}")

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
