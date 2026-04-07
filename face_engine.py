"""
Core FaceID Engine
==================
Handles face detection, embedding extraction, matching, and camera threading.
Independent of the UI.
"""

import os
import json
import time
import queue
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

# ── MediaPipe & InsightFace Imports ───────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
PERSONS_DIR      = Path("persons")
PERSONS_DIR.mkdir(exist_ok=True)
FACE_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_MODEL_PATH  = "face_detector.tflite"
EMBED_DIM        = 512
CAPTURE_POSES    = ["front", "right", "left"]

# ── Utilities ─────────────────────────────────────────────────────────────────

def list_cameras() -> list[tuple[int, str]]:
    """Detect available cameras. Returns list of (index, label)."""
    found = []
    # Linux: scan /dev/video*
    if os.path.exists("/dev"):
        video_devs = sorted(Path("/dev").glob("video*"))
        for dev in video_devs:
            try:
                idx = int(dev.name.replace("video", ""))
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    found.append((idx, f"Camera {idx} ({dev})"))
                    cap.release()
            except: pass
    # Fallback/Windows: probe indices 0..4
    if not found:
        for idx in range(5):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    found.append((idx, f"Camera {idx}"))
                    cap.release()
            except: pass
    return found if found else [(0, "Camera 0 (Default)")]

class DirectionExporter:
    """Writes recognized face coordinates to a file."""
    def __init__(self, path: Path):
        self.path = path

    def export(self, x: int, y: int, w: int, h: int, name: str, conf: float):
        cx, cy = x + w // 2, y + h // 2
        data = {
            "timestamp"  : datetime.utcnow().isoformat(),
            "name"       : name,
            "confidence" : round(float(conf), 4),
            "center"     : {"x": cx, "y": cy},
            "bbox"       : {"x": x, "y": y, "w": w, "h": h},
        }
        try:
            self.path.write_text(json.dumps(data, indent=2))
        except: pass

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b))

def load_persons() -> dict[str, list[np.ndarray]]:
    persons = {}
    if not PERSONS_DIR.exists():
        return persons
    for person_dir in sorted(PERSONS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        embeds = []
        for npy_file in sorted(person_dir.glob("*.npy")):
            try:
                emb = np.load(str(npy_file))
                embeds.append(emb)
            except Exception as e:
                print(f"[ENGINE] Error loading {npy_file}: {e}")
        if embeds:
            persons[person_dir.name] = embeds
    return persons

# ── Face Engine ───────────────────────────────────────────────────────────────

class FaceEngine:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.mp_detector = None
        self.haar = None
        self.arc = None
        self._init_mediapipe()
        self._init_arcface(use_gpu)

    def _init_mediapipe(self):
        if MEDIAPIPE_AVAILABLE:
            try:
                if not os.path.exists(FACE_MODEL_PATH):
                    print(f"[ENGINE] Downloading face detector model...")
                    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
                
                base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
                options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
                self.mp_detector = vision.FaceDetector.create_from_options(options)
                print("[ENGINE] MediaPipe Tasks FaceDetector ready")
                return
            except Exception as e:
                print(f"[ENGINE] MediaPipe Tasks init failed: {e}")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)
        print("[ENGINE] Using OpenCV Haar cascade fallback")

    def _init_arcface(self, use_gpu: bool):
        if INSIGHTFACE_AVAILABLE:
            try:
                ctx = 0 if use_gpu else -1
                self.arc = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"])
                self.arc.prepare(ctx_id=ctx, det_size=(320, 320))
                print(f"[ENGINE] ArcFace ready (GPU={use_gpu})")
            except Exception as e:
                print(f"[ENGINE] ArcFace init failed: {e}")
                self.arc = None
        else:
            print("[ENGINE] ArcFace unavailable")

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]:
        bboxes = []
        if self.mp_detector:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.mp_detector.detect(mp_image)
                if result.detections:
                    for det in result.detections:
                        bb = det.bounding_box
                        bboxes.append((int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)))
            except Exception as e:
                print(f"[ENGINE] Detection error: {e}")
        elif self.haar:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                bboxes.append((int(x), int(y), int(w), int(h)))
        return bboxes

    def get_embedding(self, frame: np.ndarray) -> np.ndarray | None:
        if self.arc:
            try:
                faces = self.arc.get(frame)
                if not faces: return None
                best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                emb = best.embedding
                return (emb / (np.linalg.norm(emb) + 1e-8)).astype(np.float32)
            except Exception as e:
                print(f"[ENGINE] Embedding error: {e}")
                return None
        return (np.random.randn(EMBED_DIM) / EMBED_DIM).astype(np.float32)

    def match(self, embedding: np.ndarray, persons: dict, threshold: float):
        best_name, best_dist = None, float("inf")
        for name, embeds in persons.items():
            for ref_emb in embeds:
                dist = cosine_distance(embedding, ref_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
        return (best_name, best_dist) if best_dist <= threshold else (None, best_dist)

    def close(self):
        if self.mp_detector:
            self.mp_detector.close()

# ── Threads ───────────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self, index, queue, w, h, fps):
        super().__init__(daemon=True)
        self.index, self.queue, self.w, self.h, self.fps = index, queue, w, h, fps
        self._stop = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret: continue
            if self.queue.full():
                try: self.queue.get_nowait()
                except: pass
            self.queue.put(frame)
        cap.release()

    def stop(self):
        self._stop.set()

class RecognitionThread(threading.Thread):
    def __init__(self, f_queue, r_queue, engine, settings, persons_ref, target_ref):
        super().__init__(daemon=True)
        self.f_queue, self.r_queue, self.engine = f_queue, r_queue, engine
        self.settings, self.persons_ref, self.target_ref = settings, persons_ref, target_ref
        self._stop = threading.Event()
        self._counter = 0
        self._last = None

    def run(self):
        while not self._stop.is_set():
            try: frame = self.f_queue.get(timeout=0.1)
            except: continue
            self._counter += 1
            if self._counter % self.settings.get("match_interval", 10) == 0:
                bboxes = self.engine.detect_faces(frame)
                matches = []
                target = self.target_ref[0]
                for (x, y, w, h) in bboxes:
                    pad = 20
                    crop = frame[max(0,y-pad):min(frame.shape[0],y+h+pad), max(0,x-pad):min(frame.shape[1],x+w+pad)]
                    if crop.size == 0: continue
                    emb = self.engine.get_embedding(crop)
                    name, dist = None, 1.0
                    if emb is not None:
                        name, dist = self.engine.match(emb, self.persons_ref, self.settings["match_threshold"])
                    conf = max(0.0, 1.0 - dist / self.settings["match_threshold"])
                    matches.append({"bbox":(x,y,w,h), "name":name, "dist":dist, "conf":conf, "is_target":(name and name==target)})
                self._last = {"frame": frame.copy(), "matches": matches}
            elif self._last:
                self._last["frame"] = frame.copy()
            
            if self._last:
                if self.r_queue.full():
                    try: self.r_queue.get_nowait()
                    except: pass
                self.r_queue.put(self._last)

    def stop(self):
        self._stop.set()

# ── Enrolment ─────────────────────────────────────────────────────────────────

class EnrolmentManager:
    """
    State machine for person enrolment. 
    UI-neutral: doesn't use cv2.imshow.
    """
    def __init__(self, engine):
        self.engine = engine
        self.poses = CAPTURE_POSES
        self.embeddings = {}
        self.current_pose_idx = 0

    def get_current_pose(self) -> str | None:
        if self.current_pose_idx < len(self.poses):
            return self.poses[self.current_pose_idx]
        return None

    def capture_pose(self, frame: np.ndarray) -> tuple[bool, str]:
        """
        Attempts to capture an embedding from the frame for the current pose.
        Returns (success, message).
        """
        pose = self.get_current_pose()
        if not pose:
            return False, "Already finished"

        bboxes = self.engine.detect_faces(frame)
        if not bboxes:
            return False, "No face detected"
        
        emb = self.engine.get_embedding(frame)
        if emb is None:
            return False, "Failed to extract embedding"
        
        self.embeddings[pose] = emb
        self.current_pose_idx += 1
        
        if self.current_pose_idx >= len(self.poses):
            return True, "All poses captured!"
        return True, f"Captured {pose}. Next: {self.get_current_pose().upper()}"

    def save(self, name: str) -> bool:
        """Saves captured embeddings to disk."""
        if len(self.embeddings) < len(self.poses):
            return False
        
        person_dir = PERSONS_DIR / name
        try:
            person_dir.mkdir(exist_ok=True)
            for pose, emb in self.embeddings.items():
                np.save(str(person_dir / f"{pose}.npy"), emb)
            return True
        except Exception as e:
            print(f"[ENROL] Save error: {e}")
            return False
