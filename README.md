AI Exam Proctor – Quick Start

Features
- Face and head pose monitoring (MediaPipe)
- Phone detection with YOLOv8 (Ultralytics)
- Voice activity detection (WebRTC VAD)
- NEW: Optional offline speech recognition (Vosk) with keyword-triggered termination
- On-screen overlays, event logging to CSV, score-based alerts/termination

Requirements
- Python 3.11 (recommended)
- Windows with a webcam and microphone
- Optional NVIDIA GPU for faster YOLO inference (CUDA)

Install (base)
- Create/activate a virtual environment.
- Install core packages: opencv-python, numpy, pandas, ultralytics, torch (CUDA variant optional), mediapipe, sounddevice, webrtcvad.

Optional: Offline Speech Recognition (Vosk)
To terminate the exam when cheating-related keywords are spoken, enable Vosk:
1) Install the Python package:
	pip install vosk
2) Model: the app can auto-download the default small English model on first run if missing.
	- To auto-download, leave CONFIG["auto_download_vosk_model"] = True (default) and run the app.
	- Otherwise, manually download from https://alphacephei.com/vosk/models and set:
	  CONFIG["vosk_model_path"] = "models/vosk-model-small-en-us-0.15"

Keyword Termination
- The list of keywords/phrases is configurable in cheatdetection.py:
  CONFIG["cheat_keywords"] = ["cheat", "answer", "google", "open book", ...]
- When Vosk recognizes any configured keyword (case-insensitive substring match), the app logs CheatKeywordDetected and terminates the exam immediately if CONFIG["terminate_on_keyword"] is True.

Running
- Activate your virtual environment and run the proctor:
  python -m cheatdetection

Controls
- Press q to quit.
- 1: Faster YOLO preset
- 2: More accurate YOLO preset

Notes
- If MediaPipe or audio packages aren’t installed, the app still runs with those features disabled (with clear warnings).
- If CUDA is available, the app auto-enables GPU and half precision for YOLO unless you override CONFIG["yolo_device"].
- Phone detection may terminate immediately (after 2-frame confirmation) when CONFIG["terminate_on_phone"] = True.

