"""
FaceID App (Professional Console)
=================================
Modern Industrial UI for FaceID system.
Uses face_engine.py for core logic.
"""

import os
import json
import time
import queue
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk

from face_engine import (
    FaceEngine, 
    GestureEngine,
    CameraThread, 
    RecognitionThread, 
    EnrolmentManager, 
    DirectionExporter,
    load_persons, 
    list_cameras,
    PERSONS_DIR
)

# ── Styling & Config ─────────────────────────────────────────────────────────
APP_TITLE      = "FaceID — Industrial Console"
DIRECTION_FILE = Path("direction")

# Modern Font Stack
FONT_MAIN = ("Segoe UI", 10)
FONT_CAM  = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_HEAD = ("Segoe UI", 9, "bold")
FONT_MONO = ("Consolas", 10) 
FONT_BIG  = ("Segoe UI", 24, "bold")
FONT_HUGE = ("Segoe UI", 32, "bold")

C = {
    "bg"        : "#0d0f14",
    "panel"     : "#151820",
    "border"    : "#2a2e3a",
    "accent"    : "#00d4ff",
    "accent2"   : "#ff6b35",
    "success"   : "#39ff14",
    "danger"    : "#ff3b3b",
    "text"      : "#e0e4f0",
    "muted"     : "#5a607a",
    "entry_bg"  : "#1e2230",
}

# Mapping: Gesture -> Command
GESTURE_CMDS = {
    "Pointing_Up": "STOP",
    "Victory": "FOLLOW",
    "Open_Palm": "STOP",
    "Thumb_Down": "BACK",
    "Thumb_Up": "FORWARD",
    "ILoveYou": "FOLLOW"
}

# ── Main Application ──────────────────────────────────────────────────────────

class FaceIDApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=C["bg"])
        self.minsize(1400, 900)

        # ── Persistent State ──────────────────────────────────────────────────
        self.settings = {
            "width": 1280, "height": 720, "fps": 30,
            "match_threshold": 0.40, "match_interval": 10,
            "use_gpu": False, "show_arrow": False,
            "show_all_bboxes": True, "show_confidence": True,
            "gesture_control": True, "captures_per_pose": 20
        }
        
        # ── Runtime State ─────────────────────────────────────────────────────
        self.engine = None
        self.gesture_engine = None
        self.cam_thread = None
        self.recog_thread = None
        self.f_queue = queue.Queue(maxsize=2)
        self.r_queue = queue.Queue(maxsize=2)
        self.exporter = DirectionExporter(DIRECTION_FILE)
        self.last_time = time.time()
        
        self.persons = load_persons()
        self.target_ref = [None]
        self.target_var = tk.StringVar(value="-- All --")
        self.cameras = list_cameras()
        self.cam_var = tk.IntVar(value=self.cameras[0][0] if self.cameras else 0)

        # ── UI Construction ───────────────────────────────────────────────────
        self._build_ui()
        self._refresh_persons()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda e: self._on_close())
        
        self._status("System Ready. Select camera and press START.")
        self._poll_results()

    # ══════════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.columnconfigure(1, weight=1) # Video
        self.rowconfigure(0, weight=1)

        # ── Left Panel (Controls) ─────────────────────────────────────────────
        left = tk.Frame(self, bg=C["panel"], width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.grid_propagate(False)

        self._header(left, "CAMERA SELECTION")
        for idx, label in self.cameras:
            rb = tk.Radiobutton(left, text=label, variable=self.cam_var, value=idx,
                                bg=C["panel"], fg=C["text"], selectcolor=C["bg"],
                                activebackground=C["panel"], activeforeground=C["accent"],
                                font=FONT_CAM)
            rb.pack(fill="x", padx=15, pady=2)

        self._divider(left)
        self._header(left, "SESSION")
        self._btn(left, "▶  START ENGINE", self._start_engine, C["accent"])
        self._btn(left, "■  STOP ENGINE",  self._stop_engine,  C["muted"])

        self._divider(left)
        self._header(left, "PERSONNEL")
        self._btn(left, "＋  ADD PERSON",    self._add_person, C["success"])
        self._btn(left, "✕  DELETE PERSON", self._del_person, C["danger"])
        
        self.listbox = tk.Listbox(left, bg=C["entry_bg"], fg=C["text"], 
                                  borderwidth=0, highlightthickness=1, 
                                  highlightbackground=C["border"], font=FONT_MAIN)
        self.listbox.pack(fill="both", expand=True, padx=15, pady=10)

        # ── Center (Video Feed) ───────────────────────────────────────────────
        center = tk.Frame(self, bg=C["bg"])
        center.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(center, bg="#05070c", highlightthickness=1, 
                                highlightbackground=C["border"])
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.create_text(500, 400, text="[ NO SIGNAL ]", fill=C["muted"], font=FONT_HUGE)

        # Hud Stats under video
        hud = tk.Frame(center, bg=C["panel"], height=50)
        hud.grid(row=1, column=0, sticky="ew", pady=(5,0))
        self.stat_fps = self._stat_lbl(hud, "FPS: --")
        self.stat_faces = self._stat_lbl(hud, "FACES: 0")
        self.stat_match = self._stat_lbl(hud, "MATCH: NONE")
        self.stat_gesture = self._stat_lbl(hud, "CMD: FOLLOW", side="right")

        # ── Right Panel (Settings) ────────────────────────────────────────────
        right = tk.Frame(self, bg=C["panel"], width=300)
        right.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        right.grid_propagate(False)

        self._header(right, "TRACKING")
        self.target_menu = ttk.Combobox(right, textvariable=self.target_var, state="readonly")
        self.target_menu.pack(padx=15, fill="x", pady=5)
        self.target_menu.bind("<<ComboboxSelected>>", self._on_target_change)

        self._divider(right)
        self._header(right, "LIVE SETTINGS")
        
        # Numeric Settings
        self._setting_row(right, "Res Width",  "width",  int)
        self._setting_row(right, "Res Height", "height", int)
        self._setting_row(right, "Match Thres","match_threshold", float)
        self._setting_row(right, "Frame Skip", "match_interval", int)
        self._setting_row(right, "Captures/Pose", "captures_per_pose", int)

        self._divider(right)
        self._header(right, "DISPLAY OPTIONS")
        self._check(right, "Show Arrow (Target)", "show_arrow")
        self._check(right, "Show All Bounding Boxes", "show_all_bboxes")
        self._check(right, "Show Confidence %", "show_confidence")
        self._check(right, "Use GPU Acceleration", "use_gpu")
        self._check(right, "Gesture Control", "gesture_control")

        # ── Status Bar ────────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(self, textvariable=self.status_var, bg=C["border"], 
                                   fg=C["accent"], anchor="w", padx=20, pady=8, font=FONT_MONO)
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")

    # ══════════════════════════════════════════════════════════════════════════
    #  UI HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _header(self, p, t):
        tk.Label(p, text=t, bg=C["panel"], fg=C["accent"], font=FONT_HEAD).pack(pady=(15, 5), anchor="w", padx=15)

    def _divider(self, p):
        tk.Frame(p, bg=C["border"], height=1).pack(fill="x", padx=10, pady=10)

    def _btn(self, p, t, cmd, col):
        btn = tk.Button(p, text=t, command=cmd, bg=C["bg"], fg=col, activebackground=C["border"],
                        activeforeground=col, relief="flat", padx=10, pady=10, font=FONT_BOLD, cursor="hand2")
        btn.pack(fill="x", padx=15, pady=4)

    def _stat_lbl(self, p, t, side="left"):
        l = tk.Label(p, text=t, bg=C["panel"], fg=C["muted"], font=FONT_CAM, padx=15, pady=5)
        l.pack(side=side)
        return l

    def _setting_row(self, p, label, key, dtype):
        row = tk.Frame(p, bg=C["panel"])
        row.pack(fill="x", padx=15, pady=3)
        tk.Label(row, text=label, bg=C["panel"], fg=C["text"], font=FONT_MAIN, width=14, anchor="w").pack(side="left")
        var = tk.StringVar(value=str(self.settings[key]))
        ent = tk.Entry(row, textvariable=var, bg=C["entry_bg"], fg=C["accent"], borderwidth=0, width=8, font=FONT_MONO)
        ent.pack(side="right")
        ent.bind("<Return>", lambda e: self._apply_setting(key, var, dtype))

    def _check(self, p, label, key):
        var = tk.BooleanVar(value=self.settings[key])
        cb = tk.Checkbutton(p, text=label, variable=var, bg=C["panel"], fg=C["text"], 
                            selectcolor=C["bg"], activebackground=C["panel"], 
                            activeforeground=C["accent"], font=FONT_MAIN,
                            command=lambda: self._apply_bool(key, var))
        cb.pack(fill="x", padx=15, pady=2)

    # ══════════════════════════════════════════════════════════════════════════
    #  LOGIC / THREADS
    # ══════════════════════════════════════════════════════════════════════════

    def _apply_setting(self, key, var, dtype):
        try:
            val = dtype(var.get())
            self.settings[key] = val
            self._status(f"Updated {key} to {val}")
        except:
            self._status(f"Invalid input for {key}", error=True)

    def _apply_bool(self, key, var):
        self.settings[key] = var.get()
        self._status(f"Toggled {key}: {self.settings[key]}")

    def _refresh_persons(self):
        self.persons = load_persons()
        self.listbox.delete(0, tk.END)
        for name in sorted(self.persons.keys()):
            self.listbox.insert(tk.END, f" {name}")
        
        current_target = self.target_var.get()
        options = ["-- All --"] + sorted(self.persons.keys())
        self.target_menu["values"] = options
        if current_target not in options:
            self.target_var.set("-- All --")
            self.target_ref[0] = None

    def _on_target_change(self, e=None):
        val = self.target_var.get()
        self.target_ref[0] = None if val == "-- All --" else val
        self._status(f"Target set to: {val}")

    def _start_engine(self):
        if self.cam_thread: self._stop_engine()
        
        cam_idx = self.cam_var.get()
        self._status(f"Initializing Engine on camera {cam_idx}...")
        
        # Re-init engine in case GPU/etc changed
        if not self.engine:
            self.engine = FaceEngine(use_gpu=self.settings["use_gpu"])
        
        if self.settings["gesture_control"] and not self.gesture_engine:
            self.gesture_engine = GestureEngine()
        
        self.f_queue = queue.Queue(maxsize=2)
        self.r_queue = queue.Queue(maxsize=2)
        
        self.cam_thread = CameraThread(cam_idx, self.f_queue, self.settings["width"], 
                                      self.settings["height"], self.settings["fps"])
        self.cam_thread.start()
        
        self.recog_thread = RecognitionThread(self.f_queue, self.r_queue, self.engine, 
                                             self.gesture_engine, self.settings, 
                                             self.persons, self.target_ref)
        self.recog_thread.start()
        self._status("Engine Running.")

    def _stop_engine(self):
        if self.cam_thread: 
            self.cam_thread.stop()
            self.cam_thread.join(timeout=1.0)
            self.cam_thread = None
        if self.recog_thread: 
            self.recog_thread.stop()
            self.recog_thread.join(timeout=1.0)
            self.recog_thread = None
        self._status("Engine Stopped.")

    def _poll_results(self):
        try:
            while True:
                res = self.r_queue.get_nowait()
                self._render(res)
        except queue.Empty: pass
        self.after(15, self._poll_results)

    def _render(self, res):
        frame = res["frame"]; matches = res["matches"]; gesture = res.get("gesture")
        drowsy = res.get("drowsy", False)
        h, w = frame.shape[:2]
        
        # Gesture Recognition logic
        current_cmd = None
        if gesture and gesture in GESTURE_CMDS:
            current_cmd = GESTURE_CMDS[gesture]
            self.stat_gesture.config(text=f"CMD: {current_cmd}", fg=C["danger"] if current_cmd=="STOP" else C["success"])
            # Display gesture on screen
            cv2.putText(frame, f"GESTURE: {gesture} -> {current_cmd}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif gesture:
            # Unknown gesture
            cv2.putText(frame, f"GESTURE: {gesture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Drowsiness Overlay
        if drowsy:
            cv2.putText(frame, "!!! DROWSY DETECTED !!!", (w//2 - 150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            self._status("DROWSINESS DETECTED!", error=True)

        matched_target = None
        for m in matches:
            x,y,bw,bh = m["bbox"]
            color = (0,255,100) if m["is_target"] else ((0,200,255) if m["name"] else (100,100,255))
            
            if not m["is_target"] and not self.settings["show_all_bboxes"]: continue
            
            cv2.rectangle(frame, (x,y), (x+bw, y+bh), color, 2)
            label = f"{m['name']}" if m["name"] else "unknown"
            if self.settings["show_confidence"] and m["name"]:
                label += f" {m['conf']:.0%}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if m["is_target"]:
                matched_target = m
                # Export with potential new command
                self.exporter.export(x, y, bw, bh, m["name"], m["conf"], command=current_cmd)

        # Arrow (Hide if STOP command is active)
        active_cmd = self.exporter.last_command
        if self.settings["show_arrow"] and matched_target and active_cmd != "STOP":
            tx, ty, tbw, tbh = matched_target["bbox"]
            cv2.arrowedLine(frame, (w//2, h//2), (tx+tbw//2, ty+tbh//2), (0,255,100), 3)

        # Draw to Canvas
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw > 10 and ch > 10:
            rgb = cv2.cvtColor(cv2.resize(frame, (cw, ch)), cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self._img_ref = img # Important for GC

        # Stats
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        fps = 1.0 / dt if dt > 0 else 0
        self.stat_fps.config(text=f"FPS: {fps:.1f}")

        self.stat_faces.config(text=f"FACES: {len(matches)}")
        if matched_target:
            self.stat_match.config(text=f"MATCH: {matched_target['name'].upper()}", fg=C["success"])
        else:
            self.stat_match.config(text="MATCH: NONE", fg=C["muted"])

    def _add_person(self):
        name = simpledialog.askstring("Identity", "Enter name (no spaces):")
        if not name: return
        name = name.strip().replace(" ", "_")
        
        was_running = self.cam_thread is not None
        self._stop_engine()
        
        win = EnrolmentWindow(self, self.engine, name, self.settings["captures_per_pose"])
        self.wait_window(win) # Wait for it to close
        
        self._refresh_persons()
        if was_running: self._start_engine()

    def _del_person(self):
        sel = self.listbox.curselection()
        if not sel: return
        name = self.listbox.get(sel[0]).strip()
        if messagebox.askyesno("Confirm", f"Wipe all records for '{name}'?"):
            shutil.rmtree(PERSONS_DIR / name, ignore_errors=True)
            self._refresh_persons()

    def _status(self, msg, error=False):
        self.status_var.set(f"[{'ERR' if error else 'LOG'}] {msg}")
        if error: self.status_bar.config(fg=C["danger"])
        else: self.status_bar.config(fg=C["accent"])

    def _on_close(self):
        self._stop_engine()
        if self.engine: self.engine.close()
        if self.gesture_engine: self.gesture_engine.close()
        self.destroy()

# ── Enrolment Window (Tkinter) ────────────────────────────────────────────────

class EnrolmentWindow(tk.Toplevel):
    def __init__(self, parent, engine, name, captures_per_pose):
        super().__init__(parent)
        self.title(f"Bio-Metric Enrolment: {name}")
        self.geometry("850x850")
        self.configure(bg=C["bg"])
        self.transient(parent)
        self.grab_set()
        
        self.parent = parent
        self.engine = engine
        self.name = name
        self.enroller = EnrolmentManager(engine, captures_per_pose)
        self.is_capturing_effect = False
        self.burst_mode = False
        
        # UI
        top = tk.Frame(self, bg=C["panel"], pady=10)
        top.pack(fill="x")
        self.info_lbl = tk.Label(top, text="PREPARING...", bg=C["panel"], fg=C["accent"], font=FONT_BIG)
        self.info_lbl.pack()
        
        self.step_lbl = tk.Label(top, text="Step 1 of 3", bg=C["panel"], fg=C["muted"], font=FONT_HEAD)
        self.step_lbl.pack()

        # Progress bar for current pose
        self.progress_frame = tk.Frame(top, bg=C["panel"])
        self.progress_frame.pack(fill="x", padx=100, pady=10)
        self.progress_canvas = tk.Canvas(self.progress_frame, bg=C["bg"], height=12, highlightthickness=0)
        self.progress_canvas.pack(fill="x")
        self.progress_rect = self.progress_canvas.create_rectangle(0, 0, 0, 12, fill=C["accent"], outline="")

        self.canvas = tk.Canvas(self, bg="#000", width=640, height=480, highlightthickness=2, highlightbackground=C["border"])
        self.canvas.pack(pady=15)
        
        btn_frame = tk.Frame(self, bg=C["bg"])
        btn_frame.pack(fill="x", pady=10)
        
        self.cap_btn = tk.Button(btn_frame, text="START BURST CAPTURE", command=self._toggle_burst, 
                                bg=C["success"], fg="#000", font=FONT_BOLD, 
                                padx=40, pady=15, relief="flat", cursor="hand2")
        self.cap_btn.pack(side="top")
        
        self.hint_lbl = tk.Label(self, text="Look directly at the camera.\nPress SPACE to start/pause burst capture.", 
                                bg=C["bg"], fg=C["text"], font=FONT_MAIN)
        self.hint_lbl.pack(pady=10)

        self.bind("<space>", lambda e: self._toggle_burst())
        self.bind("<Escape>", lambda e: self._close())
        self.protocol("WM_DELETE_WINDOW", self._close)
        
        self.cap = cv2.VideoCapture(parent.cam_var.get())
        self._update_loop()

    def _update_loop(self):
        if not self.cap or not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame.copy()
            pose = self.enroller.get_current_pose()
            if pose:
                idx = self.enroller.current_pose_idx + 1
                total_poses = len(self.enroller.poses)
                curr_cap, total_cap = self.enroller.get_progress()
                
                self.info_lbl.config(text=f"POSE: {pose.upper()}")
                self.step_lbl.config(text=f"Step {idx} of {total_poses} | Capture {curr_cap}/{total_cap}")
                
                # Update Progress Bar
                pw = self.progress_canvas.winfo_width()
                if pw > 1:
                    target_w = (curr_cap / total_cap) * pw
                    self.progress_canvas.coords(self.progress_rect, 0, 0, target_w, 10)

                if self.burst_mode:
                    self.cap_btn.config(text="BURSTING...", bg=C["accent2"])
                    # Attempt capture every frame in burst mode
                    success, msg = self.enroller.capture_pose(frame)
                    if success:
                        self.is_capturing_effect = True
                        self.after(50, self._reset_effect)
                        # If pose changed, pause burst to let user adjust
                        if "Next:" in msg or "All poses" in msg:
                            self.burst_mode = False
                else:
                    self.cap_btn.config(text=f"START {pose.upper()} BURST", bg=C["success"])

                # Update Hint
                if pose == "front": h = "Look directly at the camera."
                elif pose == "left": h = "Turn your head slightly LEFT."
                elif pose == "right": h = "Turn your head slightly RIGHT."
                else: h = "Follow the pose prompt."
                self.hint_lbl.config(text=f"{h}\nPress SPACE to begin burst.")
                
                # Overlay
                display = frame.copy()
                if not self.is_capturing_effect:
                    bboxes = self.engine.detect_faces(frame)
                    for (x,y,w,h_bb) in bboxes:
                        cv2.rectangle(display, (x,y), (x+w, y+h_bb), (0,255,100), 2)
                else:
                    # Subtle flash
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0,0), (640,480), (255,255,255), -1)
                    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.canvas.create_image(320, 240, anchor="center", image=img)
                self._img_ref = img
            else:
                self.burst_mode = False
                self.info_lbl.config(text="ENROLMENT COMPLETE")
                self.step_lbl.config(text="Saving profile...")
                self.cap_btn.config(state="disabled", text="SAVING...")
                self.update()
                if self.enroller.save(self.name):
                    total_cap = self.enroller.captures_per_pose
                    total_poses = len(self.enroller.poses)
                    messagebox.showinfo("Success", f"Bio-metrics for {self.name} archived successfully ({total_cap*total_poses} embeddings).")
                self._close()
                return

        self.after(30, self._update_loop)

    def _toggle_burst(self):
        self.burst_mode = not self.burst_mode

    def _reset_effect(self):
        self.is_capturing_effect = False

    def _close(self):
        if self.cap: self.cap.release(); self.cap = None
        self.destroy()

if __name__ == "__main__":
    FaceIDApp().mainloop()
