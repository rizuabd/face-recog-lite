import os
# Fix UI Error on Linux/Jetson
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import threading
import queue
import time
from flask import Flask, Response, render_template_string
from pyngrok import ngrok, conf

# --- COMPATIBILITY FIX (Windows vs Jetson) ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- CONFIG ---
# RTSP_URL = "rtsp://admin:idslci123@192.168.0.3:554/h264Preview_01_sub"
RTSP_URL = 0  # 0 for Webcam test
MODEL_PATH = "model/facenet_128_float32.tflite"
DB_FILE = "wajah_database.npz"
THRESHOLD = 0.8 

# --- NGROK CONFIG ---
# ⚠️ PASTE YOUR TOKEN HERE ⚠️
NGROK_AUTH_TOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE" 
PORT = 5000

# --- GLOBAL VARS ---
app = Flask(__name__)
outputFrame = None
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

# --- RECOGNITION FUNCTIONS ---
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

# --- PROCESSING THREAD ---
def detect_motion():
    global outputFrame, lock
    
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

    print("[INFO] Recognition Loop Started...")
    
    while True:
        frame = cam.read()
        if frame is None: continue

        frame_small = cv2.resize(frame, (640, 360))
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
            except Exception:
                pass

        with lock:
            outputFrame = frame_small.copy()

    cam.stop()

# --- WEB SERVER ---
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Remote Surveillance</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
            </head>
            <body style="background: #111; color: white; text-align: center; font-family: sans-serif;">
                <h2>📹 Live Face Recognition</h2>
                <img src="/video_feed" style="width: 90%; max-width: 800px; border: 2px solid #555;">
                <p>Powered by Flask & Ngrok</p>
            </body>
        </html>
    ''')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- MAIN ---
if __name__ == "__main__":
    # 1. Start Recognition Thread
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()

    # 2. Setup Ngrok
    if NGROK_AUTH_TOKEN == "PASTE_YOUR_NGROK_TOKEN_HERE":
        print("\n[ERROR] PLEASE PASTE YOUR NGROK TOKEN IN THE CODE!\n")
    else:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        # Open a HTTP tunnel on the default port 5000
        public_url = ngrok.connect(PORT).public_url
        print(f"\n[INFO] 🌍 Public URL: {public_url} \n")

    # 3. Start Flask
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)