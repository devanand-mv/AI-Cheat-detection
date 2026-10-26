"""
AI-Based Exam Cheating Detection System (Video + Audio)
Single-file runnable prototype.

"""

import cv2
import time
import threading
import queue
import csv
from datetime import datetime
import numpy as np
import math
import os
import json
import urllib.request
import zipfile
 

# --- Try to import optional modules ---
mp = None
face_detection_available = False
try:
    import mediapipe as mp
    face_detection_available = True
except Exception:
    print("Warning: mediapipe not available. Face detection will be disabled.")
    print("To enable face detection, install a mediapipe wheel compatible with your Python version (recommend Python 3.10 - 3.12).")
    mp = None

yolo_available = False
try:
    from ultralytics import YOLO
    yolo_available = True
except Exception:
    yolo_available = False

import platform

sd = None
webrtcvad = None
def _print_audio_install_instructions():
    """Print platform-specific help for installing audio/VAD dependencies."""
    system = platform.system()
    print("\n--- Audio dependencies not available ---")
    if system == 'Windows':
        print("On Windows it's easiest to install prebuilt wheels. Two options:")
        print("  1) Install wheels via pip (if compatible): pip install sounddevice webrtcvad")
        print("  2) Use pipwin to fetch Windows wheels: pip install pipwin; pipwin install sounddevice; pipwin install webrtcvad")
        print("If you see build errors for webrtcvad, ensure you have Visual C++ Build Tools installed.")
    elif system == 'Darwin':
        print("On macOS you can usually install with: pip install sounddevice webrtcvad")
        print("If sounddevice fails, you may need to install PortAudio (brew install portaudio)")
    else:
        print("On Linux: install system PortAudio dev headers, e.g. Debian/Ubuntu: sudo apt-get install portaudio19-dev")
        print("Then: pip install sounddevice webrtcvad")
    print("If you prefer to skip audio features, they are optional — the app will run without them.\n")

try:
    import sounddevice as _sd
    sd = _sd
except Exception:
    sd = None

try:
    import webrtcvad as _webrtcvad
    webrtcvad = _webrtcvad
except Exception:
    webrtcvad = None

if sd is None and webrtcvad is None:
    _print_audio_install_instructions()

vosk = None
try:
    import vosk as _vosk
    vosk = _vosk
except Exception:
    vosk = None

# -----------------------
# Configurable thresholds
# -----------------------
CONFIG = {
    "gaze_away_seconds_threshold": 3.0,     
    "absence_seconds_threshold": 3.0,       
    "multi_face_count_threshold": 2,        
    "phone_detection_interval": 0.7,        
    "speech_speech_frames_threshold": 3,    
    "audio_frame_ms": 20,                  
    "audio_device_index": None,            
    "exam_terminate_score": 50,             
    "cheating_score_threshold": 50,         
    
    "yolo_model": "yolov8n.pt",          
    "yolo_imgsz": 576,                     
    "yolo_conf": 0.20,                     
    "yolo_iou": 0.50,                     
    "yolo_max_det": 5,                     
    "yolo_device": None,                   
    "yolo_half": False,                    
    "yolo_augment": False,                 
    "phone_use_roi": True,               
    "phone_roi_vertical_expand": 2.0,      
    "phone_roi_horizontal_expand": 0.5,   
    "terminate_on_phone": True,           
    
    "enable_voice_recognition": True,     
    "voice_engine": "vosk",               
    "vosk_model_path": "models/vosk-model-small-en-us-0.15",  
    "vosk_model_url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", 
    "auto_download_vosk_model": True,     
    "cheat_keywords": [
        "cheat", "answer", "answers", "google", "search", "phone",
        "call", "whatsapp", "telegram", "chatgpt", "open book",
        "look up", "look it up", "copy", "paste", "send me",
        "message", "tell me the answer"
    ],
    "terminate_on_keyword": True,
    "gaze_away_violation_limit": 3,
    "gaze_warning_ttl_seconds": 3,
    "violation_cooldown_seconds": 3,
    "pose_ema_alpha": 0.3,
    "log_csv": "proctor_logs.csv"
}

CONFIG.setdefault("debug", False)
CONFIG.setdefault("ui_compact", True)

talking_recent_until = 0.0
last_yolo_objects = []
last_mic_rms = 0.0
last_suspicious_until = 0.0
yolo_preset = "balanced"
gpu_status = "CPU"
yolo_req_frame = None
yolo_req_roi = None
yolo_req_time = 0.0
yolo_req_pending = False
yolo_res_phone = False
yolo_res_objects = []
yolo_res_time = 0.0

asr_status = "off"            
last_asr_text = ""
cheat_keyword_detected = False
cheat_keyword_hit = ""

# -----------------------
# Utility: Logger & Event Queue
# -----------------------
event_q = queue.Queue()

def mark_suspicious(ttl_seconds: float = 10.0):
    """Mark the system as suspicious for the next ttl_seconds (for UI)."""
    try:
        globals()["last_suspicious_until"] = time.time() + float(ttl_seconds)
    except Exception:
        globals()["last_suspicious_until"] = time.time() + 10.0

def _print_audio_devices_once():
    try:
        if sd is None:
            return
        devs = sd.query_devices()
        if CONFIG.get("debug", False):
            print("[DEBUG] Audio input devices:")
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0:
                    print(f"  [{i}] {d.get('name','?')} - {d.get('max_input_channels')}ch @ default SR {d.get('default_samplerate','?')}")
    except Exception as e:
        print(f"[DEBUG] Could not list audio devices: {e}")

def log_event(event_type, details=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_q.put((ts, event_type, details))
    print(f"[{ts}] {event_type} - {details}")

def start_logger_thread(csv_path=CONFIG["log_csv"]):
    def logger_worker():
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "event_type", "details"])
        while True:
            item = event_q.get()
            if item is None:
                break
            ts, evtype, details = item
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([ts, evtype, details])
            event_q.task_done()
    t = threading.Thread(target=logger_worker, daemon=True)
    t.start()
    return t

# -----------------------
# Hardware auto-detection helpers
# -----------------------
def _auto_enable_cuda_if_available():
    """If CUDA is available and the user hasn't chosen a device, enable GPU + half precision."""
    try:
        import torch as _torch
        cuda_ok = bool(_torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if cuda_ok:
        if CONFIG.get("yolo_device") in (None, "cpu", "CPU"):
            CONFIG["yolo_device"] = "0"
        if not bool(CONFIG.get("yolo_half", False)):
            CONFIG["yolo_half"] = True
        try:
            _torch.backends.cudnn.benchmark = True
            if hasattr(_torch, 'set_float32_matmul_precision'):
                _torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        try:
            name = _torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
        globals()['gpu_status'] = f"CUDA: {name}"
        if CONFIG.get("debug", False):
            print(f"[DEBUG] CUDA detected - enabling GPU (device=0, half=True) [{name}]")
    else:
        globals()['gpu_status'] = "CPU"
        if CONFIG.get("debug", False):
            print("[DEBUG] CUDA not available - using CPU for YOLO")

# -----------------------
# Video: Mediapipe face mesh wrapper
# -----------------------
mp_face_mesh = None
mp_drawing = None
mp_face_detection = None
if face_detection_available:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection

LEFT_EYE = [33, 133]  
RIGHT_EYE = [362, 263]
NOSE_TIP = 1

def get_landmark_coords(landmarks, index, frame_w, frame_h):
    lm = landmarks[index]
    return int(lm.x * frame_w), int(lm.y * frame_h), lm.z

def estimate_head_pose(landmarks, img_w, img_h):
    try:
        pts2d = []
        indices = [33, 263, 1, 61, 291]
        for idx in indices:
            x, y, z = get_landmark_coords(landmarks, idx, img_w, img_h)
            pts2d.append((x, y))
        pts2d = np.array(pts2d, dtype=np.float64)

        pts3d = np.array([
            [-30.0,  40.0,  -30.0],  
            [ 30.0,  40.0,  -30.0],  
            [  0.0,   0.0,    0.0],  
            [-25.0, -40.0,  -25.0],  
            [ 25.0, -40.0,  -25.0],  
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w/2, img_h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))  
        success, rvec, tvec = cv2.solvePnP(pts3d, pts2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            x_angle = math.degrees(math.atan2(rmat[2,1], rmat[2,2]))
            y_angle = math.degrees(math.atan2(-rmat[2,0], sy))
            z_angle = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
        else:
            x_angle = math.degrees(math.atan2(-rmat[1,2], rmat[1,1]))
            y_angle = math.degrees(math.atan2(-rmat[2,0], sy))
            z_angle = 0
        return {"pitch": x_angle, "yaw": y_angle, "roll": z_angle}
    except Exception as e:
        return None

# -----------------------
# Optional: YOLOv8 phone detector wrapper
# -----------------------
yolo_model = None
if yolo_available:
    try:
        _auto_enable_cuda_if_available()
        yolo_model = YOLO(CONFIG.get('yolo_model', 'yolov8n.pt'))  
        if CONFIG.get("debug", False):
            print("YOLOv8 loaded for object detection.")
    except Exception as e:
        print("YOLOv8 model could not be loaded. Phone detection disabled.")
        yolo_model = None
        yolo_available = False

def detect_phone_yolo(frame, roi_frame=None):
    """Return True if phone detected in frame (if YOLO available)."""
    if (not yolo_available) or (yolo_model is None):
        return False
    source_img = roi_frame if roi_frame is not None else frame
    results = yolo_model.predict(
        source_img,
        conf=float(CONFIG.get('yolo_conf', 0.2)),
        iou=float(CONFIG.get('yolo_iou', 0.5)),
        imgsz=int(CONFIG.get('yolo_imgsz', 576)),
        classes=[67],
        max_det=int(CONFIG.get('yolo_max_det', 5)),
        device=CONFIG.get('yolo_device', None),
        half=bool(CONFIG.get('yolo_half', False)),
        augment=bool(CONFIG.get('yolo_augment', False)),
        verbose=False
    )
    global last_yolo_objects
    detected_objects = []
    for res in results:
        boxes = res.boxes
        for box, cls_id in zip(boxes.xyxy, boxes.cls):
            cls_name = res.names[int(cls_id)]
            detected_objects.append(cls_name)
            if cls_name.lower() in ['cell phone', 'cellphone', 'mobile phone', 'phone', 'cellphone', 'cell_phone', 'cell-phone']:
                if CONFIG.get("debug", False):
                    print(f"[DEBUG] PHONE DETECTED: {cls_name}")
                return True
            if 'phone' in cls_name.lower() or 'mobile' in cls_name.lower():
                if CONFIG.get("debug", False):
                    print(f"[DEBUG] PHONE DETECTED (partial match): {cls_name}")
                return True
    last_yolo_objects = detected_objects
    if detected_objects and CONFIG.get("debug", False):
        print(f"[DEBUG] YOLO detected objects: {detected_objects}")
    return False

# -----------------------
# Audio thread: VAD using webrtcvad + sounddevice
# -----------------------
class AudioVADThread(threading.Thread):
    def __init__(self, vad_mode=1):
        super().__init__(daemon=True)
        self.frame_ms = int(CONFIG.get("audio_frame_ms", 20))
        if self.frame_ms not in (10, 20, 30):
            print(f"[DEBUG] Invalid audio_frame_ms={self.frame_ms}, defaulting to 20ms")
            self.frame_ms = 20
        self.sample_rate = 16000
        self.channels = 1
        self.vad = webrtcvad.Vad(vad_mode) if webrtcvad is not None else None
        self.running = True
        self.speech_consecutive = 0
        self.last_speech_flag_time = None
        self.energy_rms_ema = None
        self.energy_consecutive = 0
        self.use_asr = False
        self.vosk_recognizer = None
        self._init_asr()

    def _init_asr(self):
        try:
            if not CONFIG.get("enable_voice_recognition", False):
                globals()["asr_status"] = "off"
                return
            if CONFIG.get("voice_engine", "vosk").lower() != "vosk":
                globals()["asr_status"] = "off"
                return
            if vosk is None:
                globals()["asr_status"] = "vosk-missing"
                print("Warning: vosk not installed. pip install vosk (optional)")
                return
            model_path = CONFIG.get("vosk_model_path", "models/vosk-model-small-en-us-0.15")
            if not os.path.isdir(model_path):
                if CONFIG.get("auto_download_vosk_model", False):
                    globals()["asr_status"] = "downloading"
                    print(f"[DEBUG] Vosk model not found at '{model_path}'. Attempting auto-download...")
                    try:
                        self._ensure_vosk_model(model_path, CONFIG.get("vosk_model_url"))
                    except Exception as de:
                        globals()["asr_status"] = "model-missing"
                        print(f"[DEBUG] Auto-download failed: {de} \nTip: Download a model from https://alphacephei.com/vosk/models and set CONFIG['vosk_model_path']")
                        return
                if not os.path.isdir(model_path):
                    globals()["asr_status"] = "model-missing"
                    print(f"Warning: Vosk model not found at '{model_path}'. Download a model and set CONFIG['vosk_model_path'].")
                    return
            model = vosk.Model(model_path)
            self.vosk_recognizer = vosk.KaldiRecognizer(model, self.sample_rate)
            self.use_asr = True
            globals()["asr_status"] = "vosk-ok"
            if CONFIG.get("debug", False):
                print(f"[DEBUG] Vosk ASR ready (model: {os.path.basename(model_path) or model_path})")
        except Exception as e:
            globals()["asr_status"] = "off"
            if CONFIG.get("debug", False):
                print(f"[DEBUG] ASR init failed: {e}")

    def _ensure_vosk_model(self, desired_model_path: str, url: str | None):
        if not url:
            raise RuntimeError("No Vosk model URL configured")
        base_dir = os.path.dirname(desired_model_path) or "."
        os.makedirs(base_dir, exist_ok=True)
        tmp_zip = os.path.join(base_dir, "vosk_model_download.zip")
        if CONFIG.get("debug", False):
            print(f"[DEBUG] Downloading Vosk model from {url} ...")
        urllib.request.urlretrieve(url, tmp_zip)
        if CONFIG.get("debug", False):
            print(f"[DEBUG] Extracting {tmp_zip} ...")
        with zipfile.ZipFile(tmp_zip, 'r') as zf:
            zf.extractall(base_dir)
        os.remove(tmp_zip)
        if not os.path.isdir(desired_model_path):
            candidates = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("vosk-model")]
            if candidates:
                real_path = os.path.join(base_dir, candidates[0])
                CONFIG["vosk_model_path"] = real_path
                if CONFIG.get("debug", False):
                    print(f"[DEBUG] Using extracted Vosk model at: {real_path}")
            else:
                raise RuntimeError("Extracted model directory not found after download")

    def run(self):
        if sd is None:
            print("Audio device not available; audio features disabled.")
            return
        frames_per_read = int(self.sample_rate * (self.frame_ms / 1000.0))
        expected_bytes = frames_per_read * self.channels * 2  
        if CONFIG.get("debug", False):
            print(f"[DEBUG] Audio thread starting: sample_rate={self.sample_rate}, frame_ms={self.frame_ms}, frames_per_read={frames_per_read}")
        try:
            device_index = CONFIG.get("audio_device_index", None)
            if device_index is not None:
                print(f"[DEBUG] Forcing audio input device index: {device_index}")
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype='int16', blocksize=frames_per_read, device=device_index) as stream:
                if CONFIG.get("debug", False):
                    print("[DEBUG] Audio stream opened successfully")
                frame_count = 0
                while self.running:
                    data, overflowed = stream.read(frames_per_read)
                    frame_count += 1
                    if overflowed:
                        if CONFIG.get("debug", False):
                            print("[DEBUG] Audio buffer overflow, skipping frame")
                        continue
                    pcm_bytes = data.tobytes()
                    if len(pcm_bytes) != expected_bytes:
                        print(f"[DEBUG] Frame size mismatch: got={len(pcm_bytes)} expected={expected_bytes}")
                        continue
                    vad_speech = False
                    try:
                        if self.vad is not None:
                            vad_speech = self.vad.is_speech(pcm_bytes, sample_rate=self.sample_rate)
                        else:
                            vad_speech = False
                    except Exception as e:
                        if CONFIG.get("debug", False):
                            print(f"[DEBUG] VAD error: {e}")
                        vad_speech = False

                    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(np.square(pcm), dtype=np.float64)) if pcm.size else 0.0)
                    globals()["last_mic_rms"] = rms
                    if self.energy_rms_ema is None:
                        self.energy_rms_ema = rms
                    alpha = 0.05
                    self.energy_rms_ema = (1 - alpha) * self.energy_rms_ema + alpha * rms
                    energy_threshold = max(150.0, self.energy_rms_ema * 2.5)  
                    energy_speech = rms > energy_threshold

                    is_speech = vad_speech or energy_speech

                    if self.use_asr and self.vosk_recognizer is not None and (vad_speech or energy_speech):
                        try:
                            if self.vosk_recognizer.AcceptWaveform(pcm_bytes):
                                res = self.vosk_recognizer.Result()
                                text = ""
                                try:
                                    text = json.loads(res).get("text", "")
                                except Exception:
                                    pass
                                if text:
                                    globals()["last_asr_text"] = text
                                    self._check_keywords(text)
                            else:
                                pres = self.vosk_recognizer.PartialResult()
                                try:
                                    ptext = json.loads(pres).get("partial", "")
                                    if ptext:
                                        globals()["last_asr_text"] = ptext
                                except Exception:
                                    pass
                        except Exception as e:
                            pass

                    if CONFIG.get("debug", False) and frame_count % 50 == 0:
                        print(f"[DEBUG] Frame {frame_count}: vad={vad_speech}, rms={rms:.1f}, ema={self.energy_rms_ema:.1f}, thr={energy_threshold:.1f}, energy={energy_speech}, consec={self.speech_consecutive}")

                    if is_speech:
                        self.speech_consecutive += 1
                        self.last_speech_flag_time = time.time()
                        if CONFIG.get("debug", False) and self.speech_consecutive == 1:
                            print(f"[DEBUG] Speech started")
                    else:
                        if CONFIG.get("debug", False) and self.speech_consecutive > 0:
                            print(f"[DEBUG] Speech ended; consec={self.speech_consecutive}")
                        self.speech_consecutive = 0
                    if self.speech_consecutive >= CONFIG["speech_speech_frames_threshold"]:
                        log_event("BackgroundSpeaking", f"Consecutive speech blocks: {self.speech_consecutive}")
                        mark_suspicious()
                        globals()["talking_recent_until"] = time.time() + 5.0
                        if CONFIG.get("debug", False):
                            print(f"[DEBUG] ALERT: BackgroundSpeaking logged!")
                        self.speech_consecutive = 0
        except Exception as e:
            print(f"Audio input stream error (continuing without audio): {e}")

    def stop(self):
        self.running = False

    def _check_keywords(self, text: str):
        try:
            text_l = text.lower()
            keywords = [k.lower() for k in CONFIG.get("cheat_keywords", [])]
            for kw in keywords:
                if kw and kw in text_l:
                    globals()["cheat_keyword_detected"] = True
                    globals()["cheat_keyword_hit"] = kw
                    log_event("CheatKeywordDetected", f"kw='{kw}' text='{text[:120]}'")
                    mark_suspicious()
                    globals()["talking_recent_until"] = time.time() + 5.0
                    break
        except Exception:
            pass

# -----------------------
# YOLO async worker
# -----------------------
class YOLOWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        global yolo_req_frame, yolo_req_roi, yolo_req_pending
        global yolo_res_phone, yolo_res_objects, yolo_res_time
        while self.running and yolo_available and (yolo_model is not None):
            if yolo_req_pending:
                frame = yolo_req_frame
                roi = yolo_req_roi
                yolo_req_pending = False
                try:
                    phone_present = detect_phone_yolo(frame, roi_frame=roi)
                    yolo_res_phone = phone_present
                    yolo_res_objects = list(last_yolo_objects) if last_yolo_objects else []
                    yolo_res_time = time.time()
                except Exception as e:
                    print("[DEBUG] YOLOWorker error:", e)
            time.sleep(0.005)

    def stop(self):
        self.running = False

# -----------------------
# Main Video + Analysis Loop
# -----------------------
def main_proctor(capture_index=0):
    start_logger_thread(CONFIG["log_csv"])

    audio_thread = None
    if sd is not None:
        _print_audio_devices_once()
        audio_thread = AudioVADThread()
        audio_thread.start()
        print("Audio VAD thread started.")
    else:
        print("Audio VAD not started (missing dependencies).")

    yolo_worker = None
    if yolo_available and (yolo_model is not None):
        yolo_worker = YOLOWorker()
        yolo_worker.start()
        print("YOLO worker thread started.")

    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    cap = cv2.VideoCapture(capture_index)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    face_mesh = None
    face_detector = None
    if mp_face_mesh is not None:
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                          max_num_faces=5,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
    if mp_face_detection is not None:
        face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    else:
        print("Face detection disabled (mediapipe not available).")

    last_seen_time = time.time()
    last_face_count = 0
    gaze_away_start = None
    last_phone_check = 0
    phone_detected_recent = False
    prev_phone_detected = False
    phone_detect_consec = 0
    last_face_bbox = None
    gaze_away_violations = 0
    last_gaze_warning_until = 0.0

    score = 0
    score_decay_per_sec = 1.0

    violation_cooldown_until = 0.0
    violation_reason = ""
    violation_pending_termination = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = None
        face_count = 0
        faces_landmarks = []
        if face_mesh is not None:
            results = face_mesh.process(rgb)
            if results and results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                faces_landmarks = results.multi_face_landmarks
        if face_detector is not None:
            try:
                det_res = face_detector.process(rgb)
                if det_res and det_res.detections:
                    face_count = max(face_count, len(det_res.detections))
                    best_area = 0
                    best_bbox = None
                    for d in det_res.detections:
                        try:
                            rb = d.location_data.relative_bounding_box
                            x1 = int(max(0, rb.xmin) * w)
                            y1 = int(max(0, rb.ymin) * h)
                            x2 = int(min(1.0, rb.xmin + rb.width) * w)
                            y2 = int(min(1.0, rb.ymin + rb.height) * h)
                            area = max(0, x2 - x1) * max(0, y2 - y1)
                            if area > best_area:
                                best_area = area
                                best_bbox = (x1, y1, x2, y2)
                        except Exception:
                            continue
                    if best_bbox is not None:
                        last_face_bbox = best_bbox
            except Exception:
                pass

        if face_count == 0:
            if time.time() - last_seen_time > CONFIG["absence_seconds_threshold"]:
                log_event("Absence", f"No face for {round(time.time()-last_seen_time,1)}s")
                mark_suspicious()
                score += 30
                last_seen_time = time.time()  
        else:
            last_seen_time = time.time()

        if face_count >= CONFIG["multi_face_count_threshold"]:
            log_event("MultiFace", f"{face_count} faces detected")
            mark_suspicious()
            score += 40

        current_yaw = 0
        current_pitch = 0
        if 'yaw_ema' not in locals():
            yaw_ema = 0.0
            pitch_ema = 0.0
            ema_init = False
        alpha = float(CONFIG.get('pose_ema_alpha', 0.3))
        if face_count >= 1:
            lm = faces_landmarks[0].landmark
            pose = estimate_head_pose(lm, w, h)
            if pose is not None:
                yaw = pose["yaw"]
                pitch = pose["pitch"]
                if not ema_init:
                    yaw_ema = yaw
                    pitch_ema = pitch
                    ema_init = True
                else:
                    yaw_ema = alpha * yaw + (1 - alpha) * yaw_ema
                    pitch_ema = alpha * pitch + (1 - alpha) * pitch_ema
                current_yaw = yaw_ema
                current_pitch = pitch_ema
                looking_away = abs(yaw) > 15 or abs(pitch) > 15
                if looking_away:
                    if CONFIG.get("debug", False):
                        print(f"[DEBUG] Looking away detected: yaw={round(yaw,1)}, pitch={round(pitch,1)}")
                    if gaze_away_start is None:
                        gaze_away_start = time.time()
                        if CONFIG.get("debug", False):
                            print(f"[DEBUG] Started gaze away timer")
                    else:
                        duration = time.time() - gaze_away_start
                        if CONFIG.get("debug", False):
                            print(f"[DEBUG] Gaze away duration: {round(duration,1)}s")
                        if duration > CONFIG["gaze_away_seconds_threshold"]:
                            gaze_away_violations += 1
                            last_gaze_warning_until = time.time() + float(CONFIG.get("gaze_warning_ttl_seconds", 3))
                            log_event("GazeAway", f"Looking away >{CONFIG['gaze_away_seconds_threshold']}s (count={gaze_away_violations}) yaw:{round(yaw,1)} pitch:{round(pitch,1)}")
                            if CONFIG.get("debug", False):
                                print(f"[DEBUG] ALERT: GazeAway event; count={gaze_away_violations}")
                            score += 15
                            if gaze_away_violations >= int(CONFIG.get("gaze_away_violation_limit", 3)) and not violation_pending_termination:
                                violation_reason = f"Looking away >{int(CONFIG['gaze_away_seconds_threshold'])}s (3 strikes)"
                                violation_cooldown_until = time.time() + float(CONFIG.get("violation_cooldown_seconds", 3))
                                violation_pending_termination = True
                            gaze_away_start = None 
                else:
                    gaze_away_start = None

        now = time.time()
        if yolo_available and (now - last_phone_check > CONFIG["phone_detection_interval"]):
            try:
                roi_frame = None
                if CONFIG.get("phone_use_roi", False) and 'last_face_bbox' in locals() and last_face_bbox is not None:
                    x1,y1,x2,y2 = last_face_bbox
                    face_h = (y2 - y1)
                    face_w = (x2 - x1)
                    vexp = float(CONFIG.get("phone_roi_vertical_expand", 2.0))
                    hexp = float(CONFIG.get("phone_roi_horizontal_expand", 0.5))
                    y2_exp = min(h, y2 + int(vexp * face_h))
                    x1e = max(0, x1 - int(hexp * face_w))
                    x2e = min(w, x2 + int(hexp * face_w))
                    if y2_exp > y1 and x2e > x1e:
                        roi_frame = frame[y1:y2_exp, x1e:x2e].copy()
                        cv2.rectangle(frame, (x1e, y1), (x2e, y2_exp), (255, 200, 0), 2)
                if not yolo_req_pending:
                    globals()['yolo_req_frame'] = frame.copy()
                    globals()['yolo_req_roi'] = roi_frame
                    globals()['yolo_req_time'] = now
                    globals()['yolo_req_pending'] = True
                last_phone_check = now
            except Exception as e:
                print("YOLO request error:", e)

        overlay = frame.copy()
        cv2.rectangle(overlay, (5,5), (520,190), (0,0,0), -1)
        wm_text = "MONITORED"
        wm_color = (60, 60, 60)
        step = 160
        for y in range(0, h, step):
            for x in range(-200, w, step):
                cv2.putText(overlay, wm_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, wm_color, 2)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, f"Faces: {face_count}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if yolo_available:
            phone_detected_recent = bool(yolo_res_phone)
            if phone_detected_recent:
                phone_detect_consec += 1
            else:
                phone_detect_consec = 0
            if CONFIG.get("terminate_on_phone", True) and phone_detect_consec >= 2 and not violation_pending_termination:
                log_event("PhoneDetected", "Phone detected (confirmed 2 frames)")
                score += 30
                violation_reason = "Phone detected"
                violation_cooldown_until = time.time() + float(CONFIG.get("violation_cooldown_seconds", 3))
                violation_pending_termination = True
            prev_phone_detected = phone_detected_recent
            cv2.putText(frame, f"Phone: {int(phone_detected_recent)}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, f"Phone: N/A", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        talking_flag = 1 if time.time() < globals().get("talking_recent_until", 0) else 0
        cv2.putText(frame, f"Talking: {talking_flag}", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        try:
            asr = globals().get("asr_status", "off")
        except Exception:
            asr = "off"
        cv2.putText(frame, f"ASR: {asr}", (300,65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,200,180), 2)
        if not CONFIG.get("ui_compact", True):
            try:
                lat = globals().get("last_asr_text", "")
                if lat:
                    show = lat[-42:]
                    cv2.putText(frame, f"Heard: {show}", (10,185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,220), 1)
            except Exception:
                pass

        score = max(0, score - score_decay_per_sec * (1/30.0))  # small decay per frame
        cv2.putText(frame, f"CheatingScore: {int(score)}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        cv2.putText(frame, f"Yaw: {int(current_yaw)} Pitch: {int(current_pitch)}", (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2)
        if not CONFIG.get("ui_compact", True):
            try:
                yolo_text = ",".join(last_yolo_objects[:4]) if last_yolo_objects else "none"
            except Exception:
                yolo_text = "none"
            cv2.putText(frame, f"YOLO: {yolo_text}", (10,155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
            try:
                cv2.putText(frame, f"Det {int(CONFIG['yolo_imgsz'])} conf {CONFIG['yolo_conf']:.2f} iou {CONFIG['yolo_iou']:.2f}", (300,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,220,180), 2)
            except Exception:
                pass
            try:
                cv2.putText(frame, f"GPU: {gpu_status}", (300,45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,220,220), 2)
            except Exception:
                pass
            try:
                mic_rms = int(globals().get("last_mic_rms", 0.0))
            except Exception:
                mic_rms = 0
            cv2.putText(frame, f"Mic: {mic_rms}", (220,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        
        cv2.putText(frame, "Press 'q' to quit", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 2)

        looking_away_now = False
        try:
            looking_away_now = abs(current_yaw) > 15 or abs(current_pitch) > 15
        except Exception:
            looking_away_now = False
        away_over_threshold_now = False
        try:
            if gaze_away_start is not None:
                away_over_threshold_now = (time.time() - gaze_away_start) > float(CONFIG.get("gaze_away_seconds_threshold", 3.0))
        except Exception:
            away_over_threshold_now = False
        reasons = []
        if phone_detected_recent:
            reasons.append("Phone")
        if face_count >= CONFIG["multi_face_count_threshold"]:
            reasons.append("Multiple faces")
        if away_over_threshold_now or (time.time() < last_gaze_warning_until):
            reasons.append(f"Looking away >{int(CONFIG.get('gaze_away_seconds_threshold',3))}s")
        suspicious_active = bool(reasons) or (time.time() < violation_cooldown_until)
        if suspicious_active:
            cv2.rectangle(frame, (w-220, 10), (w-10, 40), (0,0,255), -1)
            cv2.putText(frame, "SUSPICIOUS", (w-210, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            reason_text = violation_reason if violation_pending_termination else (", ".join(reasons) if reasons else "")
            if reason_text:
                cv2.putText(frame, f"Reason: {reason_text}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,220,255), 2)
        try:
            cv2.putText(frame, f"Gaze strikes: {gaze_away_violations}/{int(CONFIG.get('gaze_away_violation_limit',3))}", (300,85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,180,180), 2)
        except Exception:
            pass
        try:
            if gaze_away_violations > 0:
                y_pos = min(h - 20, 235)
                warn_text = f"Warning: Looked away {gaze_away_violations} time(s). Please keep your eyes on screen."
                cv2.putText(frame, warn_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        except Exception:
            pass
        if score >= CONFIG["cheating_score_threshold"]:
            cv2.rectangle(frame, (400,10), (w-10,80), (0,0,255), -1)
            cv2.putText(frame, "ALERT: Suspicious Activity", (410,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            log_event("HighSuspicion", f"Score={int(score)}")
            score -= 40

        if score >= CONFIG.get("exam_terminate_score", 100) and not violation_pending_termination:
            violation_reason = f"Cheating score >= {int(CONFIG.get('exam_terminate_score',100))}"
            violation_cooldown_until = time.time() + float(CONFIG.get("violation_cooldown_seconds", 3))
            violation_pending_termination = True

        if CONFIG.get("terminate_on_keyword", True) and globals().get("cheat_keyword_detected", False) and not violation_pending_termination:
            kw = globals().get("cheat_keyword_hit", "keyword")
            violation_reason = f"Keyword detected: {kw}"
            violation_cooldown_until = time.time() + float(CONFIG.get("violation_cooldown_seconds", 3))
            violation_pending_termination = True

        if violation_pending_termination:
            remaining = max(0, int(math.ceil(violation_cooldown_until - time.time())))
            cv2.rectangle(frame, (80, int(h/2)-60), (w-80, int(h/2)+60), (0,0,255), -1)
            cv2.putText(frame, "EXAM TERMINATING", (100, int(h/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 3)
            cv2.putText(frame, violation_reason, (100, int(h/2)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"in {remaining}s", (w-220, int(h/2)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            if time.time() >= violation_cooldown_until:
                log_event("ExamTerminated", violation_reason)
                cv2.imshow("AI Exam Proctor", frame)
                cv2.waitKey(300)
                break
        if faces_landmarks and globals().get("face_detection_available", False) and mp_drawing is not None and mp_face_mesh is not None:
            for face_lm in faces_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lm,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )
        cv2.imshow("AI Exam Proctor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('1'):
            CONFIG['yolo_imgsz'] = 384
            CONFIG['yolo_conf'] = 0.25
            CONFIG['yolo_iou'] = 0.45
            CONFIG['phone_detection_interval'] = 0.5
            globals()['yolo_preset'] = 'fast'
            log_event('YoloPresetChanged', 'fast (imgsz=384, conf=0.25, iou=0.45)')
        elif key == ord('2'):
            CONFIG['yolo_imgsz'] = 704
            CONFIG['yolo_conf'] = 0.18
            CONFIG['yolo_iou'] = 0.60
            CONFIG['phone_detection_interval'] = 0.9
            globals()['yolo_preset'] = 'accurate'
            log_event('YoloPresetChanged', 'accurate (imgsz=704, conf=0.18, iou=0.60)')

    cap.release()
    cv2.destroyAllWindows()
    if audio_thread:
        audio_thread.stop()
    event_q.put(None)
    print("Proctor stopped.")


if __name__ == "__main__":
    print("Starting AI-Based Exam Cheating Detection System")
    print("Press 'q' in window to quit.")
    main_proctor()
