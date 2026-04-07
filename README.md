# FaceID — Industrial Console
### Raspberry Pi 5 / Linux / Windows Optimized Face Recognition

A modular, high-performance face recognition system using MediaPipe Tasks for detection and ArcFace (InsightFace) for deep embeddings. Includes a professional Tkinter developer console and a headless CLI tester.

---

## 🏗️ Architecture

The system is split into three main components for maximum stability:

1.  **`face_engine.py` (Core):**
    *   **MediaPipe Tasks:** Modern face detection API (requires `face_detector.tflite`).
    *   **ArcFace:** InsightFace `buffalo_sc` model for 512-D L2-normalized embeddings.
    *   **Threading:** `CameraThread` (non-blocking capture) and `RecognitionThread` (async inference).
    *   **State Machine:** `EnrolmentManager` for UI-neutral biometric capture.

2.  **`Face_ID.py` (Tkinter UI):**
    *   Professional dark-themed industrial console.
    *   Live settings: Resolution, Threshold, Frame Skip, GPU Toggle.
    *   Integrated Enrolment Window (no external OpenCV windows).
    *   Real-time HUD and HUD export (`./direction` JSON).

3.  **`face_id_cli.py` (CLI Tester):**
    *   Headless-safe command-line interface.
    *   Ideal for remote debugging or low-resource environments.
    *   Fallback logic for systems without GUI support.

---

## 📋 Requirements

- **Python:** 3.8 - 3.11 (Recommended for maximum compatibility with MediaPipe and InsightFace).
- **OS:** Raspberry Pi OS (64-bit), Ubuntu 22.04+, or Windows 10/11.
- **Hardware:** Raspberry Pi 5 (4GB+) or a modern PC with a webcam.

---

## 🚀 Quick Start

### 1. System Dependencies (Linux/Pi)
```bash
sudo apt update
sudo apt install python3-tk python3-pip libgomp1
```

### 2. Install Python Packages
```bash
pip install -r requirements.txt
```
*Note: If you have a CUDA-capable GPU, install `onnxruntime-gpu` instead of `onnxruntime`.*

### 3. Launch
**Main Application:**
```bash
python Face_ID.py
```
**CLI Debugger:**
```bash
python face_id_cli.py
```

---

## ⚙️ Settings Reference

| Setting | Type | Description |
|---------|------|-------------|
| `Width/Height` | px | Camera capture resolution. RPi5 recommended: 320x240. |
| `Threshold` | 0-1 | Cosine distance cutoff. Lower (e.g. 0.35) is stricter. |
| `Frame Skip` | N | Runs ArcFace every N frames. Increase for better performance. |
| `GPU Toggle` | bool | Enables CUDA provider for ArcFace (requires `onnxruntime-gpu`). |

---

## 📁 Project Structure
```
.
├── face_engine.py      # Core logic, threads, and state machines
├── Face_ID.py          # Main Tkinter Application
├── face_id_cli.py      # Headless CLI Tester
├── face_detector.tflite # Downloaded automatically on first run
├── direction           # Exported JSON for the last matched target
└── persons/            # Biometric database (one folder per person)
```

---

## 🛠️ Raspberry Pi 5 Tips
- **Headless Setup:** If running via SSH, use `python face_id_cli.py`. The system will auto-detect the lack of a display and print matches to the terminal.
- **Performance:** Set `match_interval` (Frame Skip) to `10` or higher and resolution to `320x240` for 30FPS tracking on Pi 5.
- **Camera:** Compatible with USB webcams and CSI cameras via the V4L2 wrapper.
