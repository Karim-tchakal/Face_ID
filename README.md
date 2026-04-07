# FaceID — Developer Console
### Raspberry Pi 5 Optimized Face Recognition System

```
Stack: MediaPipe (detection) → ArcFace (embedding) → Cosine (matching)
UI:    Tkinter developer console with all settings exposed
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  CameraThread                                       │
│   OpenCV VideoCapture → frame_queue (maxsize=2)     │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│  RecognitionThread                                  │
│   MediaPipe detect → ArcFace embed → Cosine match  │
│   Frame skip counter (match_interval setting)       │
│   → result_queue (maxsize=2)                        │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│  FaceIDApp (Tkinter main thread)                    │
│   after(15ms) poll → render → export direction     │
└─────────────────────────────────────────────────────┘
```

---

## Install

```bash
# 1. System deps (Raspberry Pi)
sudo apt update
sudo apt install python3-tk python3-pip libgomp1

# 2. Python packages
pip install -r requirements.txt

# 3. Run
python face_id_app.py
```

---

## Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **Auto camera detect** | Scans `/dev/video*`, prompts user on startup |
| 2 | **Threading + GPU** | CameraThread + RecognitionThread; GPU via ArcFace CUDA |
| 3 | **Frame skip counter** | Configurable `match_interval` — runs recognition every N frames |
| 4 | **Tkinter UI** | Dark industrial developer console |
| 5 | **Target dropdown** | Follow one specific person or all |
| 6 | **direction file** | JSON export of matched face coords/bbox each frame |
| 7 | **Add persons** | 3-pose enrolment: front, right, left via OpenCV window |
| 8 | **Delete persons** | Remove from disk + live memory |
| 9 | **Arrow toggle** | Screen-center → recognized target face arrow overlay |
| 10 | **Clean close** | Thread join + camera release on window close |
| 11 | **Code format** | Section-commented, type-hinted, developer-friendly |
| 12 | **RPi5 target** | ARM64 notes in requirements, v4l2 camera support |
| 13 | **All settings in UI** | Resolution, FPS, threshold, frame skip, GPU toggle |

---

## direction File Format

Written to `./direction` on every frame where the target is recognized:

```json
{
  "timestamp"  : "2025-01-15T10:32:55.123456",
  "name"       : "Alice",
  "confidence" : 0.8721,
  "center"     : { "x": 320, "y": 240 },
  "bbox"       : { "x": 285, "y": 185, "w": 70, "h": 90 }
}
```

---

## Person Storage

Embeddings saved under `./persons/<name>/<pose>.npy`

```
persons/
  Alice/
    front.npy   ← ArcFace 512-d embedding, L2 normalized
    right.npy
    left.npy
  Bob/
    front.npy
    ...
```

---

## Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `width` | 640 | Camera capture width (px) |
| `height` | 480 | Camera capture height (px) |
| `fps` | 30 | Camera target FPS |
| `match_threshold` | 0.40 | Cosine distance cutoff (lower = stricter) |
| `match_interval` | 10 | Run ArcFace every N frames |
| `use_gpu` | false | Use CUDA for ArcFace inference |
| `show_all_bboxes` | true | Draw boxes around unrecognized faces too |
| `show_confidence` | true | Display confidence % on bbox label |

---

## Enrolment Flow

1. Click **＋ ADD NEW PERSON**
2. Enter name (no spaces)
3. OpenCV window opens — follow on-screen pose instructions
4. Press **SPACE** to capture each pose (front → right → left)
5. Press **Q** to abort at any time
6. Embeddings saved automatically to `./persons/`

---

## Raspberry Pi 5 Notes

- Use `libcamera-vid` with `--codec mjpeg` piped to `/dev/videoX` for CSI camera
- Or use `rpicam-apps` v4l2 output: `rpicam-vid --codec yuv420 -o - | ...`
- `mediapipe` for ARM64: check [mediapipe-rpi](https://github.com/nicedaddy/mediapipe-rpi4)
- GPU: Pi 5 has no discrete CUDA GPU — set `use_gpu = False`; ArcFace runs on CPU via ONNX Runtime
- Reduce `width/height` to 320×240 and `match_interval` to 5 for better RPi5 performance