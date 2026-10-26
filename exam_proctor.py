"""
AI-Based Exam Cheating Detection System (Video + Audio)
Single-file runnable prototype.

Features:
- Face detection + landmark analysis (mediapipe)
- Simple head/gaze estimation (approximate)
- Multi-face detection
- Optional phone/object detection using YOLOv8 (ultralytics)
- Voice activity detection using webrtcvad + sounddevice
- Alert scoring, on-screen overlay, and CSV logging

Author: (Your Name)
Date: YYYY-MM-DD
"""

import cv2
import time
import threading
import queue
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import sys
import math

# --- Try to import optional modules ---
try:
    import mediapipe as mp
except Exception as e:
    print("mediapipe is required. Install with: pip install mediapipe")
    raise

# Ultralityics / YOLO (optional)
yolo_available = False
try:
    from ultralytics import YOLO
    yolo_available = True
except Exception:
    yolo_available = False

# Audio / VAD
try:
    import sounddevice as sd
    import webrtcvad
except Exception as e:
    print("Warning: sounddevice and webrtcvad required for audio detection. Install with: pip install sounddevice webrtcvad")
    sd = None
    webrtcvad = None

# -----------------------
# Configurable thresholds
# -----------------------
CONFIG = {
    "gaze_away_seconds_threshold": 2.5,      # seconds of continuous looking away to flag
    "absence_seconds_threshold": 3.0,       # seconds missing from frame to flag
    "multi_face_count_threshold": 2,        # >1 faces flagged
    "phone_detection_interval": 2.0,        # seconds between phone detection runs
    "speech_speech_frames_threshold": 3,    # consecutive VAD positive blocks → flag
    "audio_block_duration": 1.0,            # seconds per audio capture chunk
    "cheating_score_threshold": 50,         # combined score threshold to send high alert
    "log_csv": "proctor_logs.csv"
}

# -----------------------
# Utility: Logger & Event Queue
# -----------------------
event_q = queue.Queue()

def log_event(event_type, details=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_q.put((ts, event_type, details))
    # Also print immediately
    print(f"[{ts}] {event_type} - {details}")

def start_logger_thread(csv_path=CONFIG["log_csv"]):
    def logger_worker():
        # create CSV with header if not exists
        try:
            df_existing = pd.read_csv(csv_path)
            has_header = True
        except FileNotFoundError:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "event_type", "details"])
        # Now keep consuming queue
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
# Video: Mediapipe face mesh wrapper
# -----------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Use these landmark indices for simple head/gaze heuristics
# mediapipe face mesh uses 468 landmarks; we pick eyes and nose tip
LEFT_EYE = [33, 133]   # approximate outer/inner
RIGHT_EYE = [362, 263]
NOSE_TIP = 1

def get_landmark_coords(landmarks, index, frame_w, frame_h):
    lm = landmarks[index]
    return int(lm.x * frame_w), int(lm.y * frame_h), lm.z

# Head pose estimation helper using 2D-3D solvePnP.
# Use a few key 2D points to estimate yaw/pitch roughly.
# For simplicity, assume rough 3D model points.
def estimate_head_pose(landmarks, img_w, img_h):
    # selected mediapipe indices for outer landmarks
    # left eye outer, right eye outer, nose tip, left mouth, right mouth
    try:
        pts2d = []
        indices = [33, 263, 1, 61, 291]
        for idx in indices:
            x, y, z = get_landmark_coords(landmarks, idx, img_w, img_h)
            pts2d.append((x, y))
        pts2d = np.array(pts2d, dtype=np.float64)

        # rough 3D model points in mm (approx)
        pts3d = np.array([
            [-30.0,  40.0,  -30.0],  # left eye
            [ 30.0,  40.0,  -30.0],  # right eye
            [  0.0,   0.0,    0.0],  # nose tip
            [-25.0, -40.0,  -25.0],  # left mouth
            [ 25.0, -40.0,  -25.0],  # right mouth
        ], dtype=np.float64)

        # Camera matrix approx
        focal_length = img_w
        center = (img_w/2, img_h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))  # assume no lens distortion

        success, rvec, tvec = cv2.solvePnP(pts3d, pts2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None
        # Convert rotation vector to euler angles
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
        # x_angle: rotation around x (pitch), y_angle: yaw
        return {"pitch": x_angle, "yaw": y_angle, "roll": z_angle}
    except Exception as e:
        return None

# -----------------------
# Optional: YOLOv8 phone detector wrapper
# -----------------------
yolo_model = None
if yolo_available:
    try:
        # use small model for speed
        yolo_model = YOLO('yolov8n.pt')  # will auto-download if not present
        print("YOLOv8 loaded for object detection.")
    except Exception as e:
        print("YOLOv8 model could not be loaded. Phone detection disabled.")
        yolo_model = None
        yolo_available = False

def detect_phone_yolo(frame):
    """Return True if phone detected in frame (if YOLO available)."""
    if (not yolo_available) or (yolo_model is None):
        return False
    results = yolo_model.predict(frame, conf=0.25, imgsz=640, verbose=False)
    # results is a list; use first
    for res in results:
        boxes = res.boxes
        for box, cls_id in zip(boxes.xyxy, boxes.cls):
            cls_name = res.names[int(cls_id)]
            if cls_name.lower() in ['cell phone', 'cellphone', 'mobile phone', 'phone', 'cellphone', 'cell_phone', 'cell-phone']:
                return True
            # YOLO class names can differ; also check 'handset' etc
            if 'phone' in cls_name.lower() or 'mobile' in cls_name.lower():
                return True
    return False

# -----------------------
# Audio thread: VAD using webrtcvad + sounddevice
# -----------------------
class AudioVADThread(threading.Thread):
    def __init__(self, block_duration=CONFIG["audio_block_duration"], vad_mode=2):
        super().__init__(daemon=True)
        self.block_duration = block_duration
        self.sample_rate = 16000
        self.channels = 1
        self.vad = None
        if webrtcvad is not None:
            self.vad = webrtcvad.Vad(vad_mode)
        self.running = True
        self.speech_consecutive = 0
        self.last_speech_flag_time = None

    def run(self):
        if sd is None or self.vad is None:
            print("Audio or VAD not available; audio detection disabled.")
            return
        block_size = int(self.sample_rate * self.block_duration)
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype='int16') as stream:
                while self.running:
                    data, overflowed = stream.read(block_size)
                    if overflowed:
                        # skip if buffer overflow
                        continue
                    # webrtcvad expects 16-bit mono PCM bytes
                    pcm_bytes = data.tobytes()
                    try:
                        is_speech = self.vad.is_speech(pcm_bytes, sample_rate=self.sample_rate)
                    except Exception:
                        # if frame size not appropriate for vad, skip
                        is_speech = False
                    if is_speech:
                        self.speech_consecutive += 1
                        self.last_speech_flag_time = time.time()
                    else:
                        self.speech_consecutive = 0
                    # If consecutive speech frames exceed threshold -> flag
                    if self.speech_consecutive >= CONFIG["speech_speech_frames_threshold"]:
                        log_event("BackgroundSpeaking", f"Consecutive speech blocks: {self.speech_consecutive}")
                        # reset to avoid duplicate flood
                        self.speech_consecutive = 0
                    time.sleep(0.01)
        except Exception as e:
            print("Error with audio input stream:", e)

    def stop(self):
        self.running = False

# -----------------------
# Main Video + Analysis Loop
# -----------------------
def main_proctor(capture_index=0):
    # Start logger
    start_logger_thread(CONFIG["log_csv"])

    # Start audio thread
    audio_thread = None
    if sd is not None and webrtcvad is not None:
        audio_thread = AudioVADThread()
        audio_thread.start()
        print("Audio VAD thread started.")
    else:
        print("Audio VAD not started (missing dependencies).")

    cap = cv2.VideoCapture(capture_index)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Mediapipe face mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=2,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    # State trackers
    last_seen_time = time.time()
    last_face_count = 0
    gaze_away_start = None
    last_phone_check = 0
    phone_detected_recent = False

    # Scoring for combined cheating confidence
    score = 0
    score_decay_per_sec = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Mediapipe processing
        results = face_mesh.process(rgb)
        face_count = 0
        faces_landmarks = []
        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)
            faces_landmarks = results.multi_face_landmarks

        # Presence / absence
        if face_count == 0:
            # not seen
            if time.time() - last_seen_time > CONFIG["absence_seconds_threshold"]:
                log_event("Absence", f"No face for {round(time.time()-last_seen_time,1)}s")
                score += 30
                last_seen_time = time.time()  # reset to avoid repeat logs
        else:
            last_seen_time = time.time()

        # Multiple faces
        if face_count >= CONFIG["multi_face_count_threshold"]:
            log_event("MultiFace", f"{face_count} faces detected")
            score += 40

        # Head pose / gaze check for first (primary) face
        if face_count >= 1:
            # use first face landmarks for primary candidate
            lm = faces_landmarks[0].landmark
            pose = estimate_head_pose(lm, w, h)
            if pose is not None:
                # yaw threshold: left/right looking
                yaw = pose["yaw"]
                pitch = pose["pitch"]
                # tuning: yaw > ~20 means looking aside, pitch > ~25 means looking up/down
                looking_away = abs(yaw) > 20 or abs(pitch) > 25
                if looking_away:
                    if gaze_away_start is None:
                        gaze_away_start = time.time()
                    else:
                        duration = time.time() - gaze_away_start
                        if duration > CONFIG["gaze_away_seconds_threshold"]:
                            log_event("GazeAway", f"Looking away for {round(duration,1)}s (yaw:{round(yaw,1)} pitch:{round(pitch,1)})")
                            score += 20
                            gaze_away_start = None  # reset
                else:
                    gaze_away_start = None

        # Phone detection (periodic, optional)
        now = time.time()
        if yolo_available and (now - last_phone_check > CONFIG["phone_detection_interval"]):
            try:
                phone_present = detect_phone_yolo(frame)
                last_phone_check = now
                if phone_present:
                    phone_detected_recent = True
                    log_event("PhoneDetected", "Phone or mobile device detected in frame")
                    score += 30
                else:
                    phone_detected_recent = False
            except Exception as e:
                # if YOLO fails, disable to avoid crash
                print("YOLO detection error, disabling phone detection:", e)
                global yolo_available
                yolo_available = False

        # Draw overlay and status
        overlay = frame.copy()
        # small status box
        cv2.rectangle(overlay, (5,5), (370,115), (0,0,0), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # textual statuses
        cv2.putText(frame, f"Faces: {face_count}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if yolo_available:
            cv2.putText(frame, f"Phone: {int(phone_detected_recent)}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, f"Phone: N/A", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # show audio detection if audio thread running: show "Talking" if recent event in log
        # For simplicity, if a recent BackgroundSpeaking event occurred in last 10s, show talking
        # Could be improved by sharing state variable from audio thread.
        # Here, we can check the logger file quickly (cheap) or maintain a shared flag (more efficient).
        # Use a simple recent-events check: check last few events in CSV if exists. (fast enough)
        talking_flag = False
        try:
            df_logs = pd.read_csv(CONFIG["log_csv"])
            if not df_logs.empty:
                last_row = df_logs.tail(10)
                # look for recent BackgroundSpeaking within last 10s
                now_ts = datetime.now()
                for idx, row in last_row.iterrows():
                    tstr = row['timestamp']
                    t = datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")
                    if (now_ts - t).total_seconds() < 10 and row['event_type'] == 'BackgroundSpeaking':
                        talking_flag = True
                        break
        except Exception:
            talking_flag = False

        cv2.putText(frame, f"Talking: {int(talking_flag)}", (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Show current score and warnings
        # Decay score over time
        score = max(0, score - score_decay_per_sec * (1/30.0))  # small decay per frame
        cv2.putText(frame, f"CheatingScore: {int(score)}", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # If score crosses threshold, big alert
        if score >= CONFIG["cheating_score_threshold"]:
            cv2.rectangle(frame, (400,10), (w-10,80), (0,0,255), -1)
            cv2.putText(frame, "ALERT: Suspicious Activity", (410,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            # Log high-level alert
            log_event("HighSuspicion", f"Score={int(score)}")
            # Reduce score a bit to avoid repeated logs
            score -= 40

        # Optionally draw facial landmarks for debugging
        if faces_landmarks:
            for face_lm in faces_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_lm,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                )

        # Show frame
        cv2.imshow("AI Exam Proctor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    # stop audio
    if audio_thread:
        audio_thread.stop()
    # send sentinel to logger thread to close
    event_q.put(None)
    print("Proctor stopped.")


if __name__ == "__main__":
    print("Starting AI-Based Exam Cheating Detection System")
    print("Press 'q' in window to quit.")
    main_proctor()
