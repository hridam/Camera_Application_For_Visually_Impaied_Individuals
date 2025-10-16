import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import speech_recognition as sr
import datetime
import time
import numpy as np
from gtts import gTTS
import pygame
import tempfile
import os
import sys
import math

# region -> Camera app for visually impaired users

def main():
    root = tk.Tk()
    root.title("Camera Application For Visually Impaired Individuals")
    root.geometry("880x700")

    camera_frame = ttk.Frame(root, borderwidth=2, relief="solid")
    camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
    camera_label = tk.Label(camera_frame, bg="black")
    camera_label.pack(expand=True, fill="both")

    # region -> photo feedback label diaplay
    photo_feedback = tk.StringVar(value="Say a position name to capture photo...")
    feedback_label = ttk.Label(root, textvariable=photo_feedback,
                               foreground="blue", font=("Arial", 12, "bold"), wraplength=840)
    feedback_label.pack(pady=5)

    # region -> face percentage display 
    face_percentage_var = tk.StringVar(value="Face in zone: 0%")
    percentage_label = ttk.Label(root, textvariable=face_percentage_var,
                                 foreground="darkgreen", font=("Arial", 14, "bold"))
    percentage_label.pack()

    # main_frame = ttk.Frame(root, padding=5)
    # main_frame.grid(row=0, column=0, sticky="nsew")

    # root.rowconfigure((0,1), weight=1)
    # root.columnconfigure((0,1), weight=1)

    #this for the frame and the labels 


    # region -> countdown display
    countdown_label = ttk.Label(camera_frame, text="",
                                foreground="red", font=("Arial", 72, "bold"), background="black")
    countdown_label.place(relx=0.5, rely=0.5, anchor="center")

    # option to quit button (safe shutdown)
    # control_frame = ttk.Frame(root)
    # control_frame.pack(pady=6)
    # quit_btn = ttk.Button(control_frame, text="Exit", command=lambda: safe_shutdown())
    # quit_btn.pack()

    positions = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        root.destroy()
        return

   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # region -> variables
    current_region = None
    photo_taken = False
    listening_active = True
    last_voice_time = 0.0
    face_detected = False
    target_zone = None
    current_tts_thread = None
    capture_in_progress = False
    app_running = True
    last_spoken_text = None
    last_spoken_time = 0.0

   
    message_last_time = {}
    MESSAGE_COOLDOWN_DEFAULT = 2.0  # seconds

   
    try:
        pygame.mixer.init(frequency=44100, size=-16)
    except Exception as e:
        print("pygame mixer init failed:", e)

    # region -> sounnd beep
    def generate_beep(frequency=880, duration_ms=120, volume=0.5):
        sample_rate = 44100
        n_samples = int(sample_rate * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, n_samples, False)
        waveform = np.sin(2 * np.pi * frequency * t) * volume
        waveform = (waveform * 32767).astype(np.int16)
        waveform = np.repeat(waveform[:, np.newaxis], 2, axis=1)
        try:
            return pygame.sndarray.make_sound(waveform)
        except Exception:
            
            class _NoSound:
                def play(self): pass
            return _NoSound()

    beep1 = generate_beep(880, 120)
    beep2 = generate_beep(1320, 120)

    def play_camera_beep():
        try:
            beep1.play()
            time.sleep(0.1)
            beep2.play()
        except Exception:
            pass

    def stop_speech():
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    
    tts_lock = threading.Lock()

    def speak_now(text, min_interval_same=2.0):
        """Speak text using gTTS. Avoid immediate repeats of the same text."""
        nonlocal current_tts_thread, last_spoken_text, last_spoken_time
        now = time.time()
        if text == last_spoken_text and (now - last_spoken_time) < min_interval_same:
            return
        last_spoken_text = text
        last_spoken_time = now

        def _play_tts(t):
            with tts_lock:
                try:
                    
                    stop_speech()
                    fd, path = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    gTTS(text=t, lang="en").save(path)
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and app_running:
                        time.sleep(0.08)
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                except Exception as e:
                    print("Speech error:", e)
                time.sleep(0.05)

        
        if current_tts_thread and current_tts_thread.is_alive():
            try:
                stop_speech()
                
                current_tts_thread.join(timeout=0.2)
            except Exception:
                pass

        current_tts_thread = threading.Thread(target=_play_tts, args=(text,), daemon=True)
        current_tts_thread.start()

    def speak_guarded(tag, text, cooldown=MESSAGE_COOLDOWN_DEFAULT):
        """Speak message with a tag-based cooldown to avoid spamming similar hints."""
        now = time.time()
        last = message_last_time.get(tag, 0)
        if (now - last) >= cooldown:
            message_last_time[tag] = now
            speak_now(text)

    # region -> rectangle in pixels
    def get_target_zone(region_name, frame_width, frame_height):
        third_w = frame_width / 3
        third_h = frame_height / 3
        zones = {
            "top_left": (0, 0, third_w, third_h),
            "top_right": (2 * third_w, 0, frame_width, third_h),
            "bottom_left": (0, 2 * third_h, third_w, frame_height),
            "bottom_right": (2 * third_w, 2 * third_h, frame_width, frame_height),
            "center": (third_w, third_h, 2 * third_w, 2 * third_h)
        }
        return zones.get(region_name, zones["center"])

    def get_face_center(face_rect):
        x, y, w, h = face_rect
        return (x + w // 2, y + h // 2)

    def get_zone_center(zone):
        x1, y1, x2, y2 = zone
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_face_percentage_in_zone(face_rect, zone):
        fx1, fy1, fw, fh = face_rect
        fx2, fy2 = fx1 + fw, fy1 + fh
        zx1, zy1, zx2, zy2 = zone
        ix1, iy1 = max(fx1, zx1), max(fy1, zy1)
        ix2, iy2 = min(fx2, zx2), min(fy2, zy2)
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            face_area = fw * fh
            return (intersection_area / face_area) * 100
        return 0.0

    # region -> Rotate detection
    ROTATION_ANGLES = [0, -10, 10, -15, 15]  
    def detect_faces_with_rotation(gray):
        h, w = gray.shape[:2]
        found_faces = []
        for angle in ROTATION_ANGLES:
            if angle == 0:
                search_img = gray
                M = None
                invM = None
            else:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
                search_img = rotated
                invM = cv2.invertAffineTransform(M)

            # region -> detect
            faces = face_cascade.detectMultiScale(search_img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) > 0:
                for (x, y, fw, fh) in faces:
                    if angle == 0:
                        found_faces.append((x, y, fw, fh))
                    else:
                        corners = np.array([
                            [x, y, 1],
                            [x + fw, y, 1],
                            [x, y + fh, 1],
                            [x + fw, y + fh, 1]
                        ]).T 
                        mapped = invM.dot(corners)  
                        xs = mapped[0, :]
                        ys = mapped[1, :]
                        nx1, ny1 = int(np.min(xs)), int(np.min(ys))
                        nx2, ny2 = int(np.max(xs)), int(np.max(ys))
                       
                        nx1 = max(0, nx1); ny1 = max(0, ny1)
                        nx2 = min(w - 1, nx2); ny2 = min(h - 1, ny2)
                        nfw = max(1, nx2 - nx1)
                        nfh = max(1, ny2 - ny1)
                        found_faces.append((nx1, ny1, nfw, nfh))
                
        cleaned = []
        for (x, y, fw, fh) in found_faces:
            area = fw * fh
            overlapped = False
            for i, (cx, cy, cfw, cfh) in enumerate(cleaned):
                # IoU-like check
                ix1 = max(x, cx); iy1 = max(y, cy)
                ix2 = min(x + fw, cx + cfw); iy2 = min(y + fh, cy + cfh)
                if ix1 < ix2 and iy1 < iy2:
                    # overlap -> keep the larger area
                    other_area = cfw * cfh
                    if area > other_area:
                        cleaned[i] = (x, y, fw, fh)
                    overlapped = True
                    break
            if not overlapped:
                cleaned.append((x, y, fw, fh))
        return cleaned

    def face_has_eyes(gray_frame, face_rect):
        x, y, w, h = face_rect
        pad_x = max(2, int(w * 0.06))
        pad_y = max(2, int(h * 0.06))
        rx1, ry1 = x + pad_x, y + pad_y
        rx2, ry2 = x + w - pad_x, y + h - pad_y
        if rx2 <= rx1 or ry2 <= ry1:
            return False
        face_roi = gray_frame[ry1:ry2, rx1:rx2]
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10))
        return len(eyes) >= 1  

    def capture_photo(clean_frame):
        nonlocal photo_taken, capture_in_progress, listening_active
        play_camera_beep()
        filename = f"{current_region}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, clean_frame)
        photo_feedback.set(f"Image saved: {filename}")
        
        speak_guarded("capture", "Image captured successfully.", cooldown=3.0)
        photo_taken = True
        capture_in_progress = False
        listening_active = True
        speak_guarded("next_instr", "You can now say the next position, or say quit to exit.", cooldown=3.0)

    def do_countdown_and_capture(clean_frame):
        for count in [3, 2, 1]:
            countdown_label.config(text=str(count))
            speak_now(str(count), min_interval_same=1.0)
            root.update_idletasks()
            time.sleep(1)
        countdown_label.config(text="")
        capture_photo(clean_frame)

    # region -> shutdown helper
    def safe_shutdown():
        nonlocal app_running
        app_running = False
        listening_active = False
        stop_speech()
        root.after(0, root.quit)

    # region -> main video processing loop
    def update_frame():
        nonlocal last_voice_time, face_detected, capture_in_progress, photo_taken, app_running

        if not app_running:
            return

        ret, frame = cap.read()
        if not ret:
            root.after(30, update_frame)
            return

        frame = cv2.flip(frame, 1)
        clean_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       
        if target_zone and not listening_active:
            x1, y1, x2, y2 = map(int, target_zone)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            zcx, zcy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.drawMarker(frame, (zcx, zcy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        faces = detect_faces_with_rotation(gray)
        now = time.time()

        
        displayed_pct = 0.0

        if target_zone and not listening_active and not capture_in_progress and not photo_taken:
            if len(faces) > 0:
                
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
               
                x = max(0, x); y = max(0, y)
                w = max(1, min(w, frame.shape[1] - x)); h = max(1, min(h, frame.shape[0] - y))

                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_pct = get_face_percentage_in_zone((x, y, w, h), target_zone)
                displayed_pct = face_pct  

                face_center = get_face_center((x, y, w, h))
                zone_center = get_zone_center(target_zone)

                face_size_ratio = (w * h) / (frame.shape[0] * frame.shape[1]) * 100

                
                is_front = face_has_eyes(gray, (x, y, w, h))

                if is_front:
                    if not face_detected:
                        speak_guarded("face_detected", "Face detected.", cooldown=2.5)
                        face_detected = True

                    
                    if now - last_voice_time > 1.8:
                        
                        if face_size_ratio < 3.0:
                            speak_guarded("move_closer", "Move a little closer to the camera.", cooldown=2.5)
                        elif face_size_ratio > 14.0:
                            speak_guarded("move_back", "Move a bit back from the camera.", cooldown=2.5)
                        elif face_pct >= 70.0:
                            
                            capture_in_progress = True
                            speak_guarded("perfect", "Perfect! Hold still, smile!", cooldown=3.0)
                            
                            root.after(1200, lambda cf=clean_frame.copy(): do_countdown_and_capture(cf))
                        else:
                            #
                            dx = face_center[0] - zone_center[0]
                            dy = face_center[1] - zone_center[1]
                            horiz = ""
                            vert = ""
                            if dx > 40:
                                horiz = "left"
                            elif dx < -40:
                                horiz = "right"
                            if dy > 40:
                                vert = "up"
                            elif dy < -40:
                                vert = "down"

                            if horiz or vert:
                                msg = f"Move slightly {horiz} {vert}".strip()
                                speak_guarded("move_dir", msg, cooldown=1.8)
                            else:
                               
                                hint = np.random.choice(["a little up", "a little down", "slightly left", "slightly right"])
                                speak_guarded("move_hint", f"Almost there, {hint}.", cooldown=2.5)
                        last_voice_time = now
                else:
                    if face_detected:
                        speak_guarded("no_face", "No face detected.", cooldown=2.0)
                        face_detected = False
            else:
                if face_detected:
                    speak_guarded("no_face", "No face detected.", cooldown=2.0)
                    face_detected = False

        # region -> displayed percentage
        face_percentage_var.set(f"Face in zone: {int(round(displayed_pct))}%")
        # draw the percentage on the frame
        cv2.putText(frame, f"{int(round(displayed_pct))}%", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        region_text = f"Target: {current_region.replace('_',' ') if current_region else 'none'}"
        cv2.putText(frame, region_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # region -> convert to Tk image and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        root.after(30, update_frame)

    # region -> Voice command 
    def listen_for_commands():
        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone()
        except Exception as e:
            print("Microphone init error:", e)
            speak_guarded("mic_err", "Microphone not available.", cooldown=10.0)
            return

        nonlocal current_region, listening_active, target_zone, face_detected, photo_taken, app_running

       
        speak_guarded("ready", "Camera ready. Say a position: top left, top right, bottom left, bottom right, or center. Say quit to exit.", cooldown=5.0)

        while app_running:
            if not listening_active:
                time.sleep(0.2)
                continue
            with mic as source:
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    photo_feedback.set("Listening... Say a position name.")
                    try:
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                    except sr.WaitTimeoutError:
                        continue
                except Exception as e:
                    print("Microphone read error:", e)
                    time.sleep(0.5)
                    continue

            try:
                command = recognizer.recognize_google(audio).lower()
                print("Heard:", command)

                if "quit" in command or "exit" in command:
                    speak_guarded("closing", "Closing application.", cooldown=1.0)
                   
                    root.after(0, safe_shutdown)
                    break

                matched = any(region.replace("_", " ") in command for region in positions)
                if not matched:
                    speak_guarded("not_recognized", "Command not recognized. Please say a position.", cooldown=2.0)
                else:
                    for region in positions:
                        if region.replace("_", " ") in command:
                            current_region = region
                            listening_active = False
                            face_detected = False
                            capture_in_progress = False
                            photo_taken = False
                            
                            ret, temp_frame = cap.read()
                            if ret:
                                h, w = temp_frame.shape[:2]
                                target_zone = get_target_zone(region, w, h)
                            speak_guarded("region_selected", f"{region.replace('_', ' ')} selected. Face the camera.", cooldown=3.0)
                            photo_feedback.set(f"Selected: {region.replace('_', ' ')}. Face the camera.")
                            break
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print("Speech Recognition error:", e)
                speak_guarded("sr_err", "Speech recognition service error.", cooldown=5.0)
            except Exception as e:
                print("Listener error:", e)

    listener_thread = threading.Thread(target=listen_for_commands, daemon=True)
    listener_thread.start()


     # main_frame.rowconfigure((0, 1), weight=1)
            # main_frame.columnconfigure((0,1), weight=1)


            # ---------for the design but the above loops can remake the design itslef ---------------------

            # top_left = ttk.Frame(root, borderwidth=2, relief="solid")
            # top_left.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
            # ttk.Label(top_left, text="Top Left",).pack(expand=True)

            # top_right = ttk.Frame(root, borderwidth=2, relief="solid")
            # top_right.grid(row=0, column=2, sticky="nsew", padx=0, pady=0)
            # ttk.Label(top_right, text="Top Right",).pack(expand=True)

            # bottom_left = ttk.Frame(root, borderwidth=2, relief="solid")
            # bottom_left.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
            # ttk.Label(bottom_left, text="Top Left",).pack(expand=True)

            # bottom_right = ttk.Frame(root, borderwidth=2, relief="solid")
            # bottom_right.grid(row=2, column=2, sticky="nsew", padx=0, pady=0)
            # ttk.Label(bottom_right, text="Top Left",).pack(expand=True)

            # center = ttk.Frame(root, borderwidth=2, relief="solid")
            # center.place(relx=0.5, rely=0.5, anchor="center", width=500, height=400)
            # ttk.Label(center, text="Top Left",).pack(expand=True)

            # center = ttk.Frame(root, borderwidth=2, relief="solid")
            # center.grid(row=1, column=1, sticky="nsew")
            # ttk.Label(center, text="Center").pack(expand=True)

            # ttk.Label(top_left, text="Top Left",).pack(expand=True)
            # ttk.Label(top_right, text="Top Right",).pack(expand=True)
            # ttk.Label(bottom_left, text="Top Left",).pack(expand=True)
            # ttk.Label(bottom_right, text="Top Left",).pack(expand=True)
            # ttk.Label(center, text="Top Left",).pack(expand=True)

            #--------------------------------------------------------------------------

            # root.mainloop()

    # region -> start update loop and mainloop
    update_frame()
    try:
        root.mainloop()
    finally:
        
        app_running = False
        try:
            cap.release()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        try:
            sys.exit(0)
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
