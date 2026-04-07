"""
FaceID Application
==================
Raspberry Pi 5 optimized face recognition system using:
  - MediaPipe  : Face detection & landmark extraction
  - ArcFace    : Deep embedding generation (InsightFace)
  - Cosine     : Similarity matching

Architecture:
  - CameraThread      : Captures frames in background thread
  - RecognitionThread : Runs MediaPipe + ArcFace inference (GPU if available)
  - MainApp (Tkinter) : UI, controls, settings panel
  - DirectionExporter : Writes recognized face coordinates to ./direction file

Authors  : Developer
Platform : Raspberry Pi 5 / Linux / Windows (dev)
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import json
import time
import math
import queue
import shutil
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# ── Third-Party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Optional GPU: InsightFace / ArcFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARN] insightface not installed — ArcFace disabled, using dlib/fallback")

# MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARN] mediapipe not installed")

# ── Constants ─────────────────────────────────────────────────────────────────
APP_TITLE        = "FaceID — Developer Console"
DIRECTION_FILE   = Path("direction")          # Exported coordinates target
PERSONS_DIR      = Path("persons")            # Stored face embeddings & images
PERSONS_DIR.mkdir(exist_ok=True)

MATCH_THRESHOLD  = 0.40   # Cosine distance threshold  (lower = stricter)
MATCH_INTERVAL   = 10     # Run recognition every N frames (skip frames)
EMBED_DIM        = 512    # ArcFace embedding dimension

CAPTURE_POSES    = ["front", "right", "left"]   # Enrolment poses required

# ── Color palette (dark industrial developer theme) ───────────────────────────
C = {
    "bg"        : "#0d0f14",
    "panel"     : "#151820",
    "border"    : "#2a2e3a",
    "accent"    : "#00d4ff",
    "accent2"   : "#ff6b35",
    "success"   : "#39ff14",
    "warn"      : "#ffcc00",
    "danger"    : "#ff3b3b",
    "text"      : "#e0e4f0",
    "muted"     : "#5a607a",
    "entry_bg"  : "#1e2230",
}

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two L2-normalized embedding vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a, b))


def list_cameras() -> list[tuple[int, str]]:
    """
    Detect available cameras.
    Returns list of (index, label) tuples.
    Uses v4l2 on Linux, index probing on other platforms.
    """
    found = []
    # Linux: scan /dev/video*
    if os.path.exists("/dev"):
        video_devs = sorted(Path("/dev").glob("video*"))
        for dev in video_devs:
            idx = int(dev.name.replace("video", ""))
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                label = f"Camera {idx} ({dev})"
                found.append((idx, label))
                cap.release()
    # Fallback: probe indices 0..4
    if not found:
        for idx in range(5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                found.append((idx, f"Camera {idx}"))
                cap.release()
    return found if found else [(0, "Camera 0 (default)")]


def load_persons() -> dict[str, list[np.ndarray]]:
    """
    Load all saved person embeddings from PERSONS_DIR.
    Returns { name: [embedding, ...] }
    """
    persons = {}
    for person_dir in sorted(PERSONS_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        embeds = []
        for npy_file in sorted(person_dir.glob("*.npy")):
            emb = np.load(str(npy_file))
            embeds.append(emb)
        if embeds:
            persons[person_dir.name] = embeds
    return persons


def save_embedding(name: str, pose: str, embedding: np.ndarray):
    """Save a single embedding numpy array to disk."""
    person_dir = PERSONS_DIR / name
    person_dir.mkdir(exist_ok=True)
    path = person_dir / f"{pose}.npy"
    np.save(str(path), embedding)
    print(f"[SAVE] Embedding saved → {path}")


def export_direction(x: int, y: int, w: int, h: int, name: str, confidence: float):
    """
    Write recognized face bounding-box data to the direction file.
    Format: JSON with timestamp, center, bbox, name, confidence.
    """
    cx = x + w // 2
    cy = y + h // 2
    data = {
        "timestamp"  : datetime.utcnow().isoformat(),
        "name"       : name,
        "confidence" : round(confidence, 4),
        "center"     : {"x": cx, "y": cy},
        "bbox"       : {"x": x, "y": y, "w": w, "h": h},
    }
    DIRECTION_FILE.write_text(json.dumps(data, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
#  FACE ENGINE  (MediaPipe + ArcFace)
# ══════════════════════════════════════════════════════════════════════════════

class FaceEngine:
    """
    Wraps MediaPipe face detection and ArcFace embedding extraction.
    Falls back to OpenCV Haar cascade if dependencies are missing.
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._init_mediapipe()
        self._init_arcface(use_gpu)

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_mediapipe(self):
        if MEDIAPIPE_AVAILABLE:
            mp_face = mp.solutions.face_detection
            self.mp_detector = mp_face.FaceDetection(
                model_selection=1,          # 1 = full-range model
                min_detection_confidence=0.5
            )
            print("[ENGINE] MediaPipe face detector ready")
        else:
            # Fallback: OpenCV Haar cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar = cv2.CascadeClassifier(cascade_path)
            self.mp_detector = None
            print("[ENGINE] Fallback: OpenCV Haar cascade detector")

    def _init_arcface(self, use_gpu: bool):
        if INSIGHTFACE_AVAILABLE:
            ctx = 0 if use_gpu else -1      # 0=GPU, -1=CPU
            self.arc = FaceAnalysis(
                name="buffalo_sc",          # Lightweight ArcFace model
                providers=["CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"]
            )
            self.arc.prepare(ctx_id=ctx, det_size=(320, 320))
            print(f"[ENGINE] ArcFace ready  (GPU={use_gpu})")
        else:
            self.arc = None
            print("[ENGINE] ArcFace unavailable — embeddings will be random placeholders")

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]:
        """
        Detect faces in frame.
        Returns list of (x, y, w, h) bounding boxes.
        """
        bboxes = []
        h, w = frame.shape[:2]

        if self.mp_detector:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_detector.process(rgb)
            if results.detections:
                for det in results.detections:
                    bb = det.location_data.relative_bounding_box
                    x = max(0, int(bb.xmin * w))
                    y = max(0, int(bb.ymin * h))
                    bw = int(bb.width * w)
                    bh = int(bb.height * h)
                    bboxes.append((x, y, bw, bh))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, 1.1, 4)
            for (x, y, fw, fh) in faces:
                bboxes.append((int(x), int(y), int(fw), int(fh)))

        return bboxes

    # ── Embedding ─────────────────────────────────────────────────────────────

    def get_embedding(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Extract ArcFace embedding from the largest detected face in frame.
        Returns normalized float32 ndarray of shape (512,) or None.
        """
        if self.arc:
            faces = self.arc.get(frame)
            if not faces:
                return None
            # Use the face with the largest bounding box
            best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            emb = best.embedding
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            return emb.astype(np.float32)
        else:
            # Placeholder: random unit vector (for UI testing without ArcFace)
            v = np.random.randn(EMBED_DIM).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8)

    # ── Match ─────────────────────────────────────────────────────────────────

    def match(
        self,
        embedding: np.ndarray,
        persons: dict[str, list[np.ndarray]],
        threshold: float
    ) -> tuple[str | None, float]:
        """
        Find best matching person using cosine distance.
        Returns (name, distance) or (None, best_distance).
        """
        best_name = None
        best_dist = float("inf")

        for name, embeds in persons.items():
            for ref_emb in embeds:
                dist = cosine_distance(embedding, ref_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        if best_dist <= threshold:
            return best_name, best_dist
        return None, best_dist

    def close(self):
        if self.mp_detector:
            self.mp_detector.close()


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA THREAD
# ══════════════════════════════════════════════════════════════════════════════

class CameraThread(threading.Thread):
    """
    Background thread: opens camera, pushes frames into a queue.
    Consumer: RecognitionThread
    """

    def __init__(self, camera_index: int, frame_queue: queue.Queue, settings: dict):
        super().__init__(daemon=True, name="CameraThread")
        self.camera_index = camera_index
        self.frame_queue  = frame_queue
        self.settings     = settings
        self._stop_event  = threading.Event()
        self.cap          = None

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.settings["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings["height"])
        self.cap.set(cv2.CAP_PROP_FPS,          self.settings["fps"])

        print(f"[CAM] Camera {self.camera_index} opened  "
              f"({self.settings['width']}x{self.settings['height']} "
              f"@ {self.settings['fps']}fps)")

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            # Keep queue small — drop old frames
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

        self.cap.release()
        print("[CAM] Camera thread stopped")

    def stop(self):
        self._stop_event.set()


# ══════════════════════════════════════════════════════════════════════════════
#  RECOGNITION THREAD
# ══════════════════════════════════════════════════════════════════════════════

class RecognitionThread(threading.Thread):
    """
    Background thread: pulls frames from camera queue,
    runs face detection + recognition, pushes results.

    Uses frame counter to skip matching (MATCH_INTERVAL).
    Publishes results via result_queue for the UI thread.
    """

    def __init__(
        self,
        frame_queue    : queue.Queue,
        result_queue   : queue.Queue,
        engine         : FaceEngine,
        settings       : dict,
        persons_ref    : dict,          # live reference — updated externally
        target_name_ref: list,          # [str|None] mutable single-item list
    ):
        super().__init__(daemon=True, name="RecognitionThread")
        self.frame_queue     = frame_queue
        self.result_queue    = result_queue
        self.engine          = engine
        self.settings        = settings
        self.persons_ref     = persons_ref
        self.target_name_ref = target_name_ref
        self._stop_event     = threading.Event()
        self._frame_counter  = 0
        self._last_result    = None     # Cache last recognition result

    def run(self):
        print("[RECOG] Recognition thread started")
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._frame_counter += 1
            bboxes = self.engine.detect_faces(frame)

            # Run full recognition only every MATCH_INTERVAL frames
            if self._frame_counter % self.settings["match_interval"] == 0:
                target = self.target_name_ref[0]
                match_results = []

                for (x, y, w, h) in bboxes:
                    # Crop face for embedding
                    pad = 20
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + h + pad)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size == 0:
                        continue

                    emb = self.engine.get_embedding(face_crop)
                    if emb is None:
                        continue

                    name, dist = self.engine.match(
                        emb,
                        self.persons_ref,
                        self.settings["match_threshold"]
                    )
                    confidence = max(0.0, 1.0 - dist / self.settings["match_threshold"])
                    match_results.append({
                        "bbox"      : (x, y, w, h),
                        "name"      : name,
                        "distance"  : dist,
                        "confidence": confidence,
                        "is_target" : (name is not None and name == target),
                    })

                self._last_result = {
                    "frame"  : frame.copy(),
                    "matches": match_results,
                    "bboxes" : bboxes,
                }
            else:
                # Reuse last recognition result but update frame
                if self._last_result is not None:
                    self._last_result["frame"] = frame.copy()

            if self._last_result is not None:
                # Publish to UI (drop old result if queue full)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put(self._last_result)

        print("[RECOG] Recognition thread stopped")

    def stop(self):
        self._stop_event.set()


# ══════════════════════════════════════════════════════════════════════════════
#  ENROLMENT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class EnrolmentManager:
    """
    Handles adding new persons via a 3-pose capture flow.
    Runs in the main thread but uses camera frames directly.
    """

    def __init__(self, engine: FaceEngine, camera_index: int, settings: dict):
        self.engine       = engine
        self.camera_index = camera_index
        self.settings     = settings

    def enrol(self, name: str) -> bool:
        """
        Opens an OpenCV window and guides the user through 3 poses.
        Returns True on success.
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.settings["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings["height"])

        embeddings = {}
        pose_idx   = 0
        captured   = False

        instructions = {
            "front": "Look STRAIGHT at the camera — press SPACE to capture",
            "right": "Look to the RIGHT — press SPACE to capture",
            "left" : "Look to the LEFT  — press SPACE to capture",
        }

        print(f"\n[ENROL] Starting enrolment for: {name}")

        while pose_idx < len(CAPTURE_POSES):
            pose = CAPTURE_POSES[pose_idx]
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect faces for visual feedback
            bboxes = self.engine.detect_faces(frame)
            for (x, y, w, h) in bboxes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 120), 2)

            # Overlay instructions
            instr = instructions[pose]
            cv2.putText(frame, f"Pose {pose_idx+1}/3: {pose.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
            cv2.putText(frame, instr,
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Person: {name}",
                        (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            cv2.putText(frame, "SPACE=capture  Q=abort",
                        (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(f"Enrolment — {name}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("[ENROL] Aborted by user")
                cap.release()
                cv2.destroyAllWindows()
                return False

            if key == ord(' '):
                # Capture embedding for this pose
                if not bboxes:
                    print(f"[ENROL] No face detected — try again")
                    continue

                emb = self.engine.get_embedding(frame)
                if emb is None:
                    print(f"[ENROL] Could not extract embedding — try again")
                    continue

                embeddings[pose] = emb
                print(f"[ENROL] ✓ Captured pose '{pose}'")

                # Flash green feedback
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,255,100), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, "CAPTURED!", (frame.shape[1]//2 - 80, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3)
                cv2.imshow(f"Enrolment — {name}", frame)
                cv2.waitKey(600)

                pose_idx += 1

        cap.release()
        cv2.destroyAllWindows()

        # Save all embeddings
        for pose, emb in embeddings.items():
            save_embedding(name, pose, emb)

        print(f"[ENROL] ✓ Enrolment complete for '{name}'")
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  TKINTER UI
# ══════════════════════════════════════════════════════════════════════════════

class FaceIDApp(tk.Tk):
    """
    Main Tkinter application window.
    Developer-facing: all settings exposed in the UI.
    """

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=C["bg"])
        self.resizable(True, True)
        self.minsize(1100, 700)

        # ── State ──────────────────────────────────────────────────────────
        self.settings = {
            "width"           : 640,
            "height"          : 480,
            "fps"             : 30,
            "match_threshold" : MATCH_THRESHOLD,
            "match_interval"  : MATCH_INTERVAL,
            "use_gpu"         : False,
            "show_arrow"      : False,
            "show_all_bboxes" : True,
            "show_confidence" : True,
        }

        self.cameras          = list_cameras()
        self.selected_cam_idx = tk.IntVar(value=self.cameras[0][0])
        self.persons          = load_persons()          # { name: [emb, ...] }
        self.target_name      = [None]                  # Mutable ref for threads
        self.target_var       = tk.StringVar(value="-- All --")

        # Threads & queues
        self.frame_queue   = queue.Queue(maxsize=2)
        self.result_queue  = queue.Queue(maxsize=2)
        self.cam_thread    = None
        self.recog_thread  = None
        self.engine        = None

        # Arrow toggle
        self.arrow_var     = tk.BooleanVar(value=False)

        # Tk variable mirrors for settings (for live editing)
        self._tk_vars = {}

        # ── Build UI ───────────────────────────────────────────────────────
        self._build_ui()
        self._refresh_person_list()
        self._refresh_target_menu()

        # ── Start loop ─────────────────────────────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_results()

    # ══════════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        """Construct the full Tkinter layout."""
        self.columnconfigure(0, weight=0)   # Left panel (controls)
        self.columnconfigure(1, weight=1)   # Center (video)
        self.columnconfigure(2, weight=0)   # Right panel (settings)
        self.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        self._build_status_bar()

    # ── Left Panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        left = tk.Frame(self, bg=C["panel"], width=260)
        left.grid(row=0, column=0, sticky="nsew", padx=(8,4), pady=8)
        left.grid_propagate(False)

        self._section_label(left, "CAMERA SELECT")
        for idx, label in self.cameras:
            rb = tk.Radiobutton(
                left, text=label, variable=self.selected_cam_idx, value=idx,
                bg=C["panel"], fg=C["text"], selectcolor=C["bg"],
                activebackground=C["panel"], activeforeground=C["accent"],
                font=("Courier", 9), anchor="w"
            )
            rb.pack(fill="x", padx=12, pady=1)

        self._divider(left)
        self._section_label(left, "SESSION CONTROL")

        self._btn(left, "▶  START CAMERA",  self._start_camera,  C["accent"])
        self._btn(left, "■  STOP CAMERA",   self._stop_camera,   C["muted"])

        self._divider(left)
        self._section_label(left, "TARGET TRACKING")

        tk.Label(left, text="Follow person:", bg=C["panel"], fg=C["muted"],
                 font=("Courier", 8)).pack(anchor="w", padx=12)

        self.target_menu = ttk.Combobox(
            left, textvariable=self.target_var, state="readonly",
            font=("Courier", 9), width=24
        )
        self.target_menu.pack(padx=12, pady=4, fill="x")
        self.target_menu.bind("<<ComboboxSelected>>", self._on_target_change)

        # Arrow toggle
        arrow_frame = tk.Frame(left, bg=C["panel"])
        arrow_frame.pack(fill="x", padx=12, pady=4)
        tk.Label(arrow_frame, text="Show arrow:", bg=C["panel"], fg=C["muted"],
                 font=("Courier", 8)).pack(side="left")
        self.arrow_chk = tk.Checkbutton(
            arrow_frame, variable=self.arrow_var, bg=C["panel"],
            fg=C["accent"], selectcolor=C["bg"], activebackground=C["panel"],
            command=self._on_arrow_toggle
        )
        self.arrow_chk.pack(side="left", padx=6)

        self._divider(left)
        self._section_label(left, "PERSON MANAGEMENT")

        self._btn(left, "＋  ADD NEW PERSON",    self._add_person,    C["success"])
        self._btn(left, "✕  DELETE PERSON",      self._delete_person, C["danger"])

        self._divider(left)
        self._section_label(left, "ENROLLED PERSONS")

        list_frame = tk.Frame(left, bg=C["panel"])
        list_frame.pack(fill="both", expand=True, padx=8, pady=4)

        scrollbar = tk.Scrollbar(list_frame, bg=C["border"])
        scrollbar.pack(side="right", fill="y")

        self.person_listbox = tk.Listbox(
            list_frame, bg=C["entry_bg"], fg=C["text"],
            selectbackground=C["accent"], selectforeground=C["bg"],
            font=("Courier", 9), borderwidth=0, highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.person_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.person_listbox.yview)

    # ── Center Panel (Video) ──────────────────────────────────────────────────

    def _build_center_panel(self):
        center = tk.Frame(self, bg=C["bg"])
        center.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        # Video canvas
        self.canvas = tk.Canvas(
            center, bg="#05070c", highlightthickness=1,
            highlightbackground=C["border"]
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Canvas placeholder text
        self.canvas.create_text(
            320, 240, text="[ NO FEED ]", fill=C["muted"],
            font=("Courier", 16), tags="placeholder"
        )

        # Stats bar under video
        stats_frame = tk.Frame(center, bg=C["panel"], height=30)
        stats_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.lbl_fps    = self._stat_label(stats_frame, "FPS: --")
        self.lbl_faces  = self._stat_label(stats_frame, "FACES: 0")
        self.lbl_match  = self._stat_label(stats_frame, "MATCH: none")
        self.lbl_dist   = self._stat_label(stats_frame, "DIST: --")

        self._fps_times = []

    # ── Right Panel (Settings) ────────────────────────────────────────────────

    def _build_right_panel(self):
        right = tk.Frame(self, bg=C["panel"], width=250)
        right.grid(row=0, column=2, sticky="nsew", padx=(4,8), pady=8)
        right.grid_propagate(False)

        self._section_label(right, "SETTINGS")

        params = [
            ("Resolution W",    "width",            int,   "px"),
            ("Resolution H",    "height",           int,   "px"),
            ("Camera FPS",      "fps",              int,   "fps"),
            ("Match Threshold", "match_threshold",  float, "(0-1)"),
            ("Frame Skip",      "match_interval",   int,   "frames"),
        ]

        for label, key, dtype, unit in params:
            self._setting_row(right, label, key, dtype, unit)

        self._divider(right)
        self._section_label(right, "DISPLAY OPTIONS")

        self._toggle_row(right, "Show all bboxes",  "show_all_bboxes")
        self._toggle_row(right, "Show confidence",  "show_confidence")
        self._toggle_row(right, "Use GPU (ArcFace)", "use_gpu")

        self._divider(right)
        self._section_label(right, "DIRECTION EXPORT")

        tk.Label(right, text=f"→ ./{DIRECTION_FILE}", bg=C["panel"],
                 fg=C["accent"], font=("Courier", 8)).pack(anchor="w", padx=12, pady=2)

        tk.Label(
            right,
            text="Exports center XY + bbox of\nrecognized target each frame.",
            bg=C["panel"], fg=C["muted"], font=("Courier", 8), justify="left"
        ).pack(anchor="w", padx=12)

        self._divider(right)
        self._section_label(right, "SYSTEM INFO")

        info_lines = [
            f"insightface : {'✓' if INSIGHTFACE_AVAILABLE else '✗'}",
            f"mediapipe   : {'✓' if MEDIAPIPE_AVAILABLE else '✗'}",
            f"opencv      : {cv2.__version__}",
            f"platform    : {os.uname().sysname if hasattr(os, 'uname') else 'win32'}",
        ]
        for line in info_lines:
            tk.Label(right, text=line, bg=C["panel"], fg=C["muted"],
                     font=("Courier", 8), anchor="w").pack(fill="x", padx=12, pady=1)

    # ── Status Bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="Ready — select camera and press START")
        bar = tk.Label(
            self, textvariable=self.status_var,
            bg=C["border"], fg=C["accent"], font=("Courier", 9),
            anchor="w", padx=10, pady=3
        )
        bar.grid(row=1, column=0, columnspan=3, sticky="ew")

    # ── Widget Helpers ────────────────────────────────────────────────────────

    def _section_label(self, parent, text: str):
        tk.Label(
            parent, text=text, bg=C["panel"], fg=C["accent"],
            font=("Courier", 8, "bold"), anchor="w"
        ).pack(fill="x", padx=12, pady=(10, 2))

    def _divider(self, parent):
        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", padx=8, pady=4)

    def _btn(self, parent, text: str, command, color: str):
        tk.Button(
            parent, text=text, command=command,
            bg=C["bg"], fg=color, activebackground=C["border"],
            activeforeground=color, font=("Courier", 9, "bold"),
            relief="flat", borderwidth=0, padx=10, pady=5, cursor="hand2"
        ).pack(fill="x", padx=12, pady=2)

    def _stat_label(self, parent, text: str) -> tk.Label:
        lbl = tk.Label(
            parent, text=text, bg=C["panel"], fg=C["muted"],
            font=("Courier", 8), padx=10
        )
        lbl.pack(side="left")
        return lbl

    def _setting_row(self, parent, label: str, key: str, dtype, unit: str):
        row = tk.Frame(parent, bg=C["panel"])
        row.pack(fill="x", padx=12, pady=2)

        tk.Label(row, text=label, bg=C["panel"], fg=C["text"],
                 font=("Courier", 8), width=16, anchor="w").pack(side="left")

        var = tk.StringVar(value=str(self.settings[key]))
        self._tk_vars[key] = (var, dtype)

        entry = tk.Entry(
            row, textvariable=var, width=7, bg=C["entry_bg"],
            fg=C["accent"], insertbackground=C["accent"],
            font=("Courier", 8), relief="flat", bd=2
        )
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda e, k=key: self._apply_setting(k))

        tk.Label(row, text=unit, bg=C["panel"], fg=C["muted"],
                 font=("Courier", 7)).pack(side="left")

    def _toggle_row(self, parent, label: str, key: str):
        var = tk.BooleanVar(value=self.settings[key])
        self._tk_vars[key] = (var, bool)

        row = tk.Frame(parent, bg=C["panel"])
        row.pack(fill="x", padx=12, pady=1)

        tk.Checkbutton(
            row, text=label, variable=var,
            bg=C["panel"], fg=C["text"], selectcolor=C["bg"],
            activebackground=C["panel"], activeforeground=C["accent"],
            font=("Courier", 8), anchor="w",
            command=lambda k=key, v=var: self._apply_bool_setting(k, v)
        ).pack(side="left")

    # ══════════════════════════════════════════════════════════════════════════
    #  SETTINGS APPLICATION
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_setting(self, key: str):
        var, dtype = self._tk_vars[key]
        try:
            self.settings[key] = dtype(var.get())
            self._status(f"Setting '{key}' = {self.settings[key]}")
        except ValueError:
            self._status(f"Invalid value for '{key}'", error=True)

    def _apply_bool_setting(self, key: str, var: tk.BooleanVar):
        self.settings[key] = var.get()
        self._status(f"Setting '{key}' = {self.settings[key]}")

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMERA / THREAD CONTROL
    # ══════════════════════════════════════════════════════════════════════════

    def _start_camera(self):
        if self.cam_thread and self.cam_thread.is_alive():
            self._status("Camera already running", error=True)
            return

        # Apply any pending settings from entry widgets
        for key in list(self._tk_vars.keys()):
            self._apply_setting(key)

        cam_idx = self.selected_cam_idx.get()

        # (Re)create engine — use_gpu may have changed
        if self.engine:
            self.engine.close()
        self.engine = FaceEngine(use_gpu=self.settings["use_gpu"])

        # Reload persons
        self.persons = load_persons()
        self._refresh_person_list()
        self._refresh_target_menu()

        # Fresh queues
        self.frame_queue  = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)

        self.cam_thread = CameraThread(cam_idx, self.frame_queue, self.settings)
        self.cam_thread.start()

        self.recog_thread = RecognitionThread(
            self.frame_queue, self.result_queue,
            self.engine, self.settings,
            self.persons, self.target_name
        )
        self.recog_thread.start()

        self._status(f"Camera {cam_idx} started — running recognition")

    def _stop_camera(self):
        if self.cam_thread:
            self.cam_thread.stop()
        if self.recog_thread:
            self.recog_thread.stop()
        self._status("Camera stopped")

    # ══════════════════════════════════════════════════════════════════════════
    #  RESULT POLLING (runs in Tk main thread via after())
    # ══════════════════════════════════════════════════════════════════════════

    def _poll_results(self):
        try:
            result = self.result_queue.get_nowait()
            self._render_result(result)
        except queue.Empty:
            pass
        self.after(15, self._poll_results)     # ~66 fps poll rate

    def _render_result(self, result: dict):
        frame   = result["frame"].copy()
        matches = result["matches"]
        h, w    = frame.shape[:2]

        # ── FPS ──────────────────────────────────────────────────────────────
        now = time.monotonic()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        fps = len(self._fps_times)

        # ── Draw detections ───────────────────────────────────────────────────
        matched_target = None

        for m in matches:
            x, y, bw, bh = m["bbox"]
            name     = m["name"]
            dist     = m["distance"]
            conf     = m["confidence"]
            is_match = name is not None
            is_target = m["is_target"]

            # Colour coding
            if is_target:
                color = (0, 255, 100)       # Green — matched target
            elif is_match:
                color = (0, 200, 255)       # Cyan  — matched but not target
            elif self.settings["show_all_bboxes"]:
                color = (100, 100, 255)     # Blue  — unrecognized face
            else:
                continue

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)

            # Label
            label = name if name else "unknown"
            if self.settings["show_confidence"] and is_match:
                label += f"  {conf:.0%}"

            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            if is_target:
                matched_target = m
                # Export coordinates
                export_direction(x, y, bw, bh, name, conf)

        # ── Arrow overlay ─────────────────────────────────────────────────────
        if self.settings["show_arrow"] and matched_target:
            cx_s = w // 2
            cy_s = h // 2
            x, y, bw, bh = matched_target["bbox"]
            cx_t = x + bw // 2
            cy_t = y + bh // 2

            cv2.arrowedLine(frame, (cx_s, cy_s), (cx_t, cy_t),
                            (0, 255, 100), 3, tipLength=0.15)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS:{fps}  FACES:{len(matches)}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Canvas render ─────────────────────────────────────────────────────
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            canvas_w, canvas_h = 640, 480

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (canvas_w, canvas_h))

        from PIL import Image, ImageTk
        img = Image.fromarray(frame_resized)
        self._photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # ── Stats labels ──────────────────────────────────────────────────────
        self.lbl_fps.config(text=f"FPS: {fps}")
        self.lbl_faces.config(text=f"FACES: {len(matches)}")

        if matched_target:
            name = matched_target["name"]
            conf = matched_target["confidence"]
            self.lbl_match.config(text=f"MATCH: {name}", fg=C["success"])
            self.lbl_dist.config(text=f"CONF: {conf:.0%}")
        else:
            self.lbl_match.config(text="MATCH: none", fg=C["muted"])
            self.lbl_dist.config(text="DIST: --")

    # ══════════════════════════════════════════════════════════════════════════
    #  PERSON MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _add_person(self):
        """Prompt for name then run 3-pose enrolment."""
        name = simpledialog.askstring(
            "Add Person", "Enter name (no spaces):",
            parent=self
        )
        if not name:
            return
        name = name.strip().replace(" ", "_")
        if not name:
            return

        was_running = self.cam_thread and self.cam_thread.is_alive()
        if was_running:
            self._stop_camera()
            time.sleep(0.3)     # Let threads stop

        enroller = EnrolmentManager(
            self.engine or FaceEngine(self.settings["use_gpu"]),
            self.selected_cam_idx.get(),
            self.settings
        )
        success = enroller.enrol(name)

        if success:
            self.persons = load_persons()
            self._refresh_person_list()
            self._refresh_target_menu()
            self._status(f"Person '{name}' enrolled successfully")

            if was_running:
                self._start_camera()
        else:
            self._status("Enrolment cancelled or failed", error=True)

    def _delete_person(self):
        """Delete selected person from listbox."""
        sel = self.person_listbox.curselection()
        if not sel:
            self._status("No person selected in the list", error=True)
            return

        name = self.person_listbox.get(sel[0])
        confirm = messagebox.askyesno(
            "Delete Person",
            f"Delete all data for '{name}'?\nThis cannot be undone.",
            parent=self
        )
        if not confirm:
            return

        person_dir = PERSONS_DIR / name
        if person_dir.exists():
            shutil.rmtree(str(person_dir))
            print(f"[DEL] Deleted person: {name}")

        self.persons = load_persons()

        # Reset target if it was the deleted person
        if self.target_name[0] == name:
            self.target_name[0] = None
            self.target_var.set("-- All --")

        self._refresh_person_list()
        self._refresh_target_menu()
        self._status(f"Person '{name}' deleted")

    # ── List & Menu Refresh ───────────────────────────────────────────────────

    def _refresh_person_list(self):
        self.person_listbox.delete(0, tk.END)
        for name in sorted(self.persons.keys()):
            n_embeds = len(self.persons[name])
            self.person_listbox.insert(tk.END, f"{name}  [{n_embeds} embed]")

    def _refresh_target_menu(self):
        options = ["-- All --"] + sorted(self.persons.keys())
        self.target_menu["values"] = options
        # Preserve current selection if still valid
        current = self.target_var.get()
        if current not in options:
            self.target_var.set("-- All --")
            self.target_name[0] = None

    def _on_target_change(self, _event=None):
        val = self.target_var.get()
        if val == "-- All --":
            self.target_name[0] = None
            self._status("Tracking all recognized faces")
        else:
            self.target_name[0] = val
            self._status(f"Tracking target: {val}")

    def _on_arrow_toggle(self):
        self.settings["show_arrow"] = self.arrow_var.get()
        state = "ON" if self.settings["show_arrow"] else "OFF"
        self._status(f"Direction arrow: {state}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STATUS & CLOSE
    # ══════════════════════════════════════════════════════════════════════════

    def _status(self, msg: str, error: bool = False):
        self.status_var.set(f"{'[ERR] ' if error else ''}{msg}")
        print(f"[UI] {msg}")

    def _on_close(self):
        """Clean shutdown: stop threads, release camera, destroy window."""
        print("[APP] Shutting down...")
        self._stop_camera()

        # Give threads a moment to exit
        if self.cam_thread:
            self.cam_thread.join(timeout=1.5)
        if self.recog_thread:
            self.recog_thread.join(timeout=1.5)

        if self.engine:
            self.engine.close()

        cv2.destroyAllWindows()
        self.destroy()
        print("[APP] Goodbye.")


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA SELECTION DIALOG  (shown on startup)
# ══════════════════════════════════════════════════════════════════════════════

def ask_camera_selection(cameras: list[tuple[int, str]]) -> int | None:
    """
    Show a simple Tk dialog asking the user which camera to use.
    Returns the selected camera index, or None to cancel.
    """
    root = tk.Tk()
    root.title("Select Camera")
    root.configure(bg=C["bg"])
    root.resizable(False, False)

    result = {"index": None}

    tk.Label(
        root, text="Detected Cameras", bg=C["bg"], fg=C["accent"],
        font=("Courier", 12, "bold"), pady=10
    ).pack()

    var = tk.IntVar(value=cameras[0][0])
    for idx, label in cameras:
        tk.Radiobutton(
            root, text=label, variable=var, value=idx,
            bg=C["bg"], fg=C["text"], selectcolor=C["panel"],
            activebackground=C["bg"], activeforeground=C["accent"],
            font=("Courier", 10), anchor="w"
        ).pack(fill="x", padx=30, pady=3)

    def confirm():
        result["index"] = var.get()
        root.destroy()

    def cancel():
        root.destroy()

    btn_frame = tk.Frame(root, bg=C["bg"])
    btn_frame.pack(pady=15)

    tk.Button(
        btn_frame, text="Use Selected Camera", command=confirm,
        bg=C["accent"], fg=C["bg"], font=("Courier", 10, "bold"),
        relief="flat", padx=20, pady=6
    ).pack(side="left", padx=8)

    tk.Button(
        btn_frame, text="Cancel", command=cancel,
        bg=C["border"], fg=C["text"], font=("Courier", 10),
        relief="flat", padx=10, pady=6
    ).pack(side="left")

    root.mainloop()
    return result["index"]


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Detect cameras and ask user which to use
    cameras = list_cameras()
    print(f"[INIT] Detected cameras: {cameras}")

    if len(cameras) == 1:
        selected_cam = cameras[0][0]
        print(f"[INIT] Single camera detected — auto-selecting camera {selected_cam}")
    else:
        selected_cam = ask_camera_selection(cameras)
        if selected_cam is None:
            print("[INIT] No camera selected — exiting")
            exit(0)

    # Launch main app
    app = FaceIDApp()
    app.selected_cam_idx.set(selected_cam)
    app.mainloop()