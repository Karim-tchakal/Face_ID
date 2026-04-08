"""
FaceID CLI Tester
================
Simple command-line interface to test enrolment and recognition logic.
"""

import sys
import time
import cv2
import queue
import numpy as np
from face_engine import FaceEngine, EnrolmentManager, load_persons, CameraThread, RecognitionThread

def main_menu():
    print("\n--- FaceID CLI Tester ---")
    print("1. Enrol New Person")
    print("2. Run Live Recognition")
    print("3. List Enrolled Persons")
    print("q. Exit")
    return input("Choice: ").strip().lower()

def safe_imshow(win_name, frame):
    """Try to show frame, print warning if cv2 is headless."""
    try:
        cv2.imshow(win_name, frame)
        return True
    except cv2.error as e:
        if "not implemented" in str(e):
            print("\n[WARN] OpenCV GUI not available (headless). Cannot show windows.")
            print("Try: pip uninstall opencv-python-headless && pip install opencv-python")
            return False
        raise e

def do_enrol(engine):
    name = input("Enter name for enrolment: ").strip()
    if not name: return
    cam_idx = int(input("Enter camera index (default 0): ") or 0)
    caps_per_pose = int(input("Enter captures per pose (default 20): ") or 20)
    
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"FAILED: Could not open camera {cam_idx}")
        return

    enroller = EnrolmentManager(engine, captures_per_pose=caps_per_pose)
    print(f"\nStarting enrolment for {name} ({caps_per_pose} images per pose).")
    print("Instructions: Press SPACE in the video window to capture. Keep holding/pressing for burst.")
    print("Poses: FRONT -> RIGHT -> LEFT")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            pose = enroller.get_current_pose()
            if not pose: break
            
            display = frame.copy()
            bboxes = engine.detect_faces(frame)
            for (x, y, w, h) in bboxes:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            curr, total = enroller.get_progress()
            cv2.putText(display, f"Pose: {pose.upper()} ({curr}/{total})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "SPACE: Capture | Q: Abort", (10, display.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if not safe_imshow(f"Enrolment: {name}", display):
                # Headless fallback: just wait for a face and auto-capture
                if bboxes:
                    success, msg = enroller.capture_pose(frame)
                    print(f"Auto-capture {pose}: {msg}")
                    if not enroller.get_current_pose(): break
                time.sleep(0.1)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                print("Aborted.")
                return
            if key == ord(' '):
                success, msg = enroller.capture_pose(frame)
                if success:
                    print(f"Progress: {msg}")
                    # Brief flash
                    flash = np.full_like(display, (255, 255, 255))
                    cv2.addWeighted(flash, 0.2, display, 0.8, 0, display)
                    cv2.imshow(f"Enrolment: {name}", display)
                    cv2.waitKey(10)
    finally:
        cap.release()
        try: cv2.destroyAllWindows()
        except: pass

    if enroller.save(name):
        print(f"\nSUCCESS: {name} enrolled and saved to disk.")
    else:
        print("\nFAILED: Enrolment incomplete or save error.")

def do_recog(engine):
    persons = load_persons()
    if not persons:
        print("No persons enrolled yet.")
        return
    
    cam_idx = int(input("Enter camera index (default 0): ") or 0)
    f_q = queue.Queue(maxsize=2)
    r_q = queue.Queue(maxsize=2)
    
    settings = {"match_threshold": 0.4, "match_interval": 5}
    target_ref = [None]
    
    cam = CameraThread(cam_idx, f_q, 640, 480, 30)
    # 4th arg is gesture_engine, use None for CLI
    rec = RecognitionThread(f_q, r_q, engine, None, settings, persons, target_ref)
    
    cam.start()
    rec.start()
    
    print("Running recognition. Press 'q' in the video window to stop.")
    try:
        while True:
            try: result = r_q.get(timeout=1)
            except: continue
            
            frame = result["frame"]
            for m in result["matches"]:
                x, y, w, h = m["bbox"]
                label = f"{m['name']} ({m['conf']:.2f})" if m['name'] else "Unknown"
                color = (0, 255, 0) if m['name'] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if not safe_imshow("Recognition Test (CLI)", frame):
                print("Headless mode: Recognition results (last 1s):")
                for m in result["matches"]:
                    print(f" - Found: {m['name'] or 'Unknown'} (Conf: {m['conf']:.2f})")
                time.sleep(1)
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cam.stop()
        rec.stop()
        cam.join(1)
        rec.join(1)
        try: cv2.destroyAllWindows()
        except: pass

def list_persons():
    persons = load_persons()
    print("\nEnrolled Persons:")
    for name in persons:
        print(f" - {name} ({len(persons[name])} embeddings)")

if __name__ == "__main__":
    engine = FaceEngine(use_gpu=False)
    while True:
        try:
            choice = main_menu()
            if choice == '1': do_enrol(engine)
            elif choice == '2': do_recog(engine)
            elif choice == '3': list_persons()
            elif choice == 'q': break
            else: print("Invalid choice.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
