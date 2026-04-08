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
FACE_MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/face_detector/face_detector_full_range/float16/1/face_detector_full_range.tflite"
FACE_MODEL_PATH  = "face_detector_full_range.tflite"
LANDMARK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
LANDMARK_MODEL_PATH = "face_landmarker.task"
GESTURE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
GESTURE_MODEL_PATH = "gesture_recognizer.task"
EMBED_DIM        = 512
CAPTURE_POSES    = ["front", "left", "right"]

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
        self.last_command = "FOLLOW" # Default state

    def export(self, x: int, y: int, w: int, h: int, name: str, conf: float, command: str = None):
        if command:
            self.last_command = command
            
        cx, cy = x + w // 2, y + h // 2
        data = {
            "timestamp"  : datetime.utcnow().isoformat(),
            "name"       : name,
            "confidence" : round(float(conf), 4),
            "center"     : {"x": cx, "y": cy},
            "bbox"       : {"x": x, "y": y, "w": w, "h": h},
            "command"    : self.last_command
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
        self.mp_landmarker = None
        self.haar = None
        self.arc = None
        self._init_mediapipe()
        self._init_face_landmarker()
        self._init_arcface(use_gpu)

    def _init_mediapipe(self):
        if MEDIAPIPE_AVAILABLE:
            try:
                if not os.path.exists(FACE_MODEL_PATH):
                    print(f"[ENGINE] Downloading face detector model...")
                    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
                
                base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
                options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.4)
                self.mp_detector = vision.FaceDetector.create_from_options(options)
                print("[ENGINE] MediaPipe Tasks FaceDetector ready")
            except Exception as e:
                print(f"[ENGINE] MediaPipe Tasks init failed: {e}")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)
        print("[ENGINE] Using OpenCV Haar cascade fallback")

    def _init_face_landmarker(self):
        if MEDIAPIPE_AVAILABLE:
            try:
                if not os.path.exists(LANDMARK_MODEL_PATH):
                    print(f"[ENGINE] Downloading face landmarker model...")
                    urllib.request.urlretrieve(LANDMARK_MODEL_URL, LANDMARK_MODEL_PATH)
                
                base_options = python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True
                )
                self.mp_landmarker = vision.FaceLandmarker.create_from_options(options)
                print("[ENGINE] MediaPipe FaceLandmarker ready")
            except Exception as e:
                print(f"[ENGINE] FaceLandmarker init failed: {e}")

    def detect_drowsiness(self, frame: np.ndarray) -> bool:
        """Uses blendshapes (EyeBlink) from FaceLandmarker to detect closed eyes."""
        if not self.mp_landmarker: return False
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.mp_landmarker.detect(mp_image)
            if result.face_blendshapes:
                # blendshapes is a list of lists (one per face)
                # each element is a Category object with category_name and score
                for face_bs in result.face_blendshapes:
                    blink_l = next((c.score for c in face_bs if c.category_name == "eyeBlinkLeft"), 0)
                    blink_r = next((c.score for c in face_bs if c.category_name == "eyeBlinkRight"), 0)
                    # Score is 0 to 1, where 1 is fully closed
                    if blink_l > 0.5 and blink_r > 0.5:
                        return True
        except Exception as e:
            print(f"[ENGINE] Drowsiness detection error: {e}")
        return False

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

# ── Gesture Engine ────────────────────────────────────────────────────────────

class GestureEngine:
    def __init__(self):
        self.recognizer = None
        self._init_gesture_recognizer()

    def _init_gesture_recognizer(self):
        if MEDIAPIPE_AVAILABLE:
            try:
                if not os.path.exists(GESTURE_MODEL_PATH):
                    print(f"[ENGINE] Downloading gesture recognizer model...")
                    urllib.request.urlretrieve(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
                
                base_options = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)
                options = vision.GestureRecognizerOptions(base_options=base_options)
                self.recognizer = vision.GestureRecognizer.create_from_options(options)
                print("[ENGINE] MediaPipe GestureRecognizer ready")
            except Exception as e:
                print(f"[ENGINE] MediaPipe GestureRecognizer init failed: {e}")

    def detect_gesture(self, frame: np.ndarray) -> str | None:
        if not self.recognizer: return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.recognizer.recognize(mp_image)
            if result.gestures:
                # Get the highest confidence gesture
                top_gesture = result.gestures[0][0]
                return top_gesture.category_name
        except Exception as e:
            print(f"[ENGINE] Gesture detection error: {e}")
        return None

    def close(self):
        if self.recognizer:
            self.recognizer.close()

# ── Threads ───────────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self, index, queue, w, h, fps):
        super().__init__(daemon=True)
        self.index, self.queue, self.w, self.h, self.fps = index, queue, w, h, fps
        self._stop_event = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.01)
                continue
            
            # Non-blocking put to avoid hanging if consumer stops
            try:
                if self.queue.full():
                    self.queue.get_nowait()
                self.queue.put(frame, timeout=0.1)
            except (queue.Full, queue.Empty):
                pass
                
        cap.release()

    def stop(self):
        self._stop_event.set()

class RecognitionThread(threading.Thread):
    def __init__(self, f_queue, r_queue, engine, gesture_engine, settings, persons_ref, target_ref):
        super().__init__(daemon=True)
        self.f_queue, self.r_queue, self.engine = f_queue, r_queue, engine
        self.gesture_engine = gesture_engine
        self.settings, self.persons_ref, self.target_ref = settings, persons_ref, target_ref
        self._stop_event = threading.Event()
        self._counter = 0
        self._last = None

    def run(self):
        while not self._stop_event.is_set():
            try: 
                frame = self.f_queue.get(timeout=0.1)
            except queue.Empty: 
                continue
                
            self._counter += 1
            if self._counter % self.settings.get("match_interval", 10) == 0:
                bboxes = self.engine.detect_faces(frame)
                matches = []
                target = self.target_ref[0]
                for (x, y, w, h) in bboxes:
                    pad = 20
                    h_img, w_img = frame.shape[:2]
                    y1, y2 = max(0, y-pad), min(h_img, y+h+pad)
                    x1, x2 = max(0, x-pad), min(w_img, x+w+pad)
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size == 0: continue
                    emb = self.engine.get_embedding(crop)
                    name, dist = None, 1.0
                    if emb is not None:
                        name, dist = self.engine.match(emb, self.persons_ref, self.settings["match_threshold"])
                    conf = max(0.0, 1.0 - dist / self.settings["match_threshold"])
                    matches.append({"bbox":(x,y,w,h), "name":name, "dist":dist, "conf":conf, "is_target":(name and name==target)})
                
                gesture = self.gesture_engine.detect_gesture(frame) if self.gesture_engine else None
                drowsy = self.engine.detect_drowsiness(frame)
                self._last = {"frame": frame.copy(), "matches": matches, "gesture": gesture, "drowsy": drowsy}
            elif self._last:
                self._last["frame"] = frame.copy()
            
            if self._last:
                try:
                    if self.r_queue.full():
                        self.r_queue.get_nowait()
                    self.r_queue.put(self._last, timeout=0.1)
                except (queue.Full, queue.Empty):
                    pass

    def stop(self):
        self._stop_event.set()

# ── Enrolment ─────────────────────────────────────────────────────────────────

class EnrolmentManager:
    """
    State machine for person enrolment. 
    UI-neutral: doesn't use cv2.imshow.
    """
    def __init__(self, engine, captures_per_pose: int = 20):
        self.engine = engine
        self.poses = CAPTURE_POSES
        self.captures_per_pose = captures_per_pose
        self.embeddings = {pose: [] for pose in self.poses}
        self.current_pose_idx = 0

    def get_current_pose(self) -> str | None:
        if self.current_pose_idx < len(self.poses):
            return self.poses[self.current_pose_idx]
        return None

    def get_progress(self) -> tuple[int, int]:
        """Returns (current_capture_count, total_needed_for_this_pose)."""
        pose = self.get_current_pose()
        if not pose: return (0, 0)
        return (len(self.embeddings[pose]), self.captures_per_pose)

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
        
        self.embeddings[pose].append(emb)
        
        count = len(self.embeddings[pose])
        if count >= self.captures_per_pose:
            self.current_pose_idx += 1
            next_pose = self.get_current_pose()
            if not next_pose:
                return True, "All poses captured!"
            return True, f"Captured {pose}. Next: {next_pose.upper()}"
        
        return True, f"Captured {pose} ({count}/{self.captures_per_pose})"

    def save(self, name: str) -> bool:
        """Saves captured embeddings to disk."""
        # Ensure at least one capture per pose
        for pose in self.poses:
            if not self.embeddings[pose]:
                return False
        
        person_dir = PERSONS_DIR / name
        try:
            person_dir.mkdir(exist_ok=True)
            for pose, embs in self.embeddings.items():
                for i, emb in enumerate(embs):
                    np.save(str(person_dir / f"{pose}_{i}.npy"), emb)
            return True
        except Exception as e:
            print(f"[ENROL] Save error: {e}")
            return False
