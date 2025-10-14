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


def main():
    root = tk.Tk()
    root.title("Camera Application For Visually Impaied Individuals")
    root.geometry("800x650")

     # main_frame = ttk.Frame(root, padding=5)
    # main_frame.grid(row=0, column=0, sticky="nsew")

    # root.rowconfigure((0,1), weight=1)
    # root.columnconfigure((0,1), weight=1)

    #this for the frame and the labels 


    camera_frame = ttk.Frame(root, borderwidth=2, relief="solid")
    camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
    camera_label = tk.Label(camera_frame, bg="black")
    camera_label.pack(expand=True, fill="both")

    photo_feedback = tk.StringVar(value="Say a position name to capture photo...")
    feedback_label = ttk.Label(root, textvariable=photo_feedback,
                               foreground="blue", font=("Arial", 12, "bold"), wraplength=750)
    feedback_label.pack(pady=5)

    countdown_label = ttk.Label(camera_frame, text="",
                                foreground="red", font=("Arial", 48, "bold"), background="black")
    countdown_label.place(relx=0.5, rely=0.5, anchor="center")

    positions = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    
    current_region = None
    photo_taken = False
    listening_active = True
    last_voice_time = 0
    face_detected = False
    target_zone = None
    current_tts_thread = None
    capture_in_progress = False
    app_running = True
    last_spoken_text = None
    last_spoken_time = 0

    pygame.mixer.init(frequency=44100, size=-16)

    # sound if the images has been captured.------------------------------------
    def generate_beep(frequency=880, duration_ms=150, volume=0.5):
        sample_rate = 44100
        n_samples = int(sample_rate * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, n_samples, False)
        waveform = np.sin(2 * np.pi * frequency * t) * volume
        waveform = (waveform * 32767).astype(np.int16)
        waveform = np.repeat(waveform[:, np.newaxis], 2, axis=1)
        return pygame.sndarray.make_sound(waveform)

    beep1 = generate_beep(880, 120)
    beep2 = generate_beep(1320, 120)

    def play_camera_beep():
        beep1.play()
        time.sleep(0.1)
        beep2.play()

    # -------------- Speech -------------------
    def stop_speech():
        if pygame.mixer.get_busy():
            pygame.mixer.stop()

    def speak_now(text, delay=0.4):
        """Avoid repeating same text within 5 seconds."""
        nonlocal current_tts_thread, last_spoken_text, last_spoken_time
        now = time.time()
        if text == last_spoken_text and (now - last_spoken_time) < 5:
            return
        last_spoken_text = text
        last_spoken_time = now

        def play_tts(t):
            stop_speech()
            try:
                fd, path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                gTTS(text=t, lang="en").save(path)
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                os.remove(path)
            except Exception as e:
                print("Speech error:", e)
            time.sleep(delay)

        if current_tts_thread and current_tts_thread.is_alive():
            stop_speech()
        current_tts_thread = threading.Thread(target=play_tts, args=(text,), daemon=True)
        current_tts_thread.start()

    
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
        return 0

    
    def capture_photo(clean_frame):
        nonlocal photo_taken, capture_in_progress, listening_active
        play_camera_beep()
        filename = f"{current_region}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, clean_frame)
        photo_feedback.set("Image captured successfully.")
        speak_now("Image captured successfully.")
        photo_taken = True
        capture_in_progress = False
        listening_active = True
        speak_now("You can now say the next position or say quit or exit to exit the application.")

    def do_countdown_and_capture(clean_frame):
        for count in [3, 2, 1]:
            countdown_label.config(text=str(count))
            speak_now(str(count))
            root.update()
            time.sleep(1)
        countdown_label.config(text="")
        capture_photo(clean_frame)

    # this is the main frame things to control the faces and detection ----------------------
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
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))
        now = time.time()

        if target_zone and not listening_active:
            x1, y1, x2, y2 = target_zone
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        if not listening_active and target_zone and not capture_in_progress and not photo_taken:
            if len(faces) > 0:
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_pct = get_face_percentage_in_zone(face, target_zone)
                face_center = get_face_center(face)
                zone_center = get_zone_center(target_zone)

                face_size_ratio = (w * h) / (frame.shape[0] * frame.shape[1]) * 100

                if not face_detected:
                    speak_now("Face detected.")
                    face_detected = True

                if now - last_voice_time > 2:
                    dx, dy = face_center[0] - zone_center[0], face_center[1] - zone_center[1]

                    # Distance feedback
                    if face_size_ratio < 3:
                        speak_now("Move a little closer to the camera.")
                    elif face_size_ratio > 12:
                        speak_now("Move a bit back from the camera.")
                    elif face_pct > 75:
                        capture_in_progress = True
                        speak_now("Perfect! Hold still, smile!")
                        root.after(1500, lambda cf=clean_frame: do_countdown_and_capture(cf))
                    else:
                        horizontal = ""
                        vertical = ""
                        if dx > 50:
                            horizontal = "left"
                        elif dx < -50:
                            horizontal = "right"
                        if dy > 50:
                            vertical = "up"
                        elif dy < -50:
                            vertical = "down"

                        if horizontal or vertical:
                            speak_now(f"Move slightly {horizontal} {vertical}".strip())
                        else:
                            hint = np.random.choice([
                                "a little up", "a bit down", "slightly to the left", "slightly to the right"
                            ])
                            speak_now(f"Almost there, {hint}.")
                    last_voice_time = now
            else:
                if face_detected:
                    speak_now("No face detected.")
                    face_detected = False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
        root.after(30, update_frame)

            # if ret:
            #     frame = cv2.resize(frame, (320, 240))
            #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     img = Image.fromarray(frame_rgb)
            #     imgtk = ImageTk.PhotoImage(image=img)

            #     for lbl in lables.values():
            #         lbl.imgtk = imgtk
            #         lbl.configure(image=imgtk)

            # root.after(30, update_frame)
        
        # update_frame()

    # All voice Commands and everything that are being spoken in this applicaiton -------------------
    def listen_for_commands():
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        nonlocal current_region, listening_active, target_zone, face_detected, photo_taken, app_running, capture_in_progress

        speak_now("Camera ready. Say a position: top left, top right, bottom left, bottom right, or center. Say quit or exit to exit.")

        while app_running:
            if not listening_active:
                time.sleep(1)
                continue
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                photo_feedback.set("Listening... Say a position name.")
                try:
                    audio = recognizer.listen(source, timeout=5)
                except sr.WaitTimeoutError:
                    continue

            try:
                command = recognizer.recognize_google(audio).lower()
                print("Heard:", command)

                if "quit" in command or "exit" in command:
                    speak_now("Closing application.")
                    stop_speech()
                    app_running = False
                    root.destroy()
                    break

                matched = any(region.replace("_", " ") in command for region in positions)
                if not matched:
                    speak_now("Command not recognized. Please say a position.")
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
                            speak_now(f"{region.replace('_', ' ')} selected. Face the camera.")
                            break
            except sr.UnknownValueError:
                print("Could not understand audio.")
            except sr.RequestError as e:
                print("Speech Recognition error:", e)
                speak_now("Speech recognition error.")



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

    threading.Thread(target=listen_for_commands, daemon=True).start()
    update_frame()
    root.mainloop()
    cap.release()
    pygame.mixer.quit()
    sys.exit()


if __name__ == "__main__":
    main()
