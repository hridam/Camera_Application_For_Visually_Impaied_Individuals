import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import speech_recognition as sr 
import datetime
import os
import time 

def main():
    root = tk.Tk()
    root.title("Voice-Controlled Auto-Capture")
    root.geometry("900x600")

    # main_frame = ttk.Frame(root, padding=5)
    # main_frame.grid(row=0, column=0, sticky="nsew")

    # root.rowconfigure((0,1), weight=1)
    # root.columnconfigure((0,1), weight=1)

    #this for the frame and the labels 


    for r in range(3):
        root.rowconfigure(r, weight=1)
    for c in range(3):
        root.columnconfigure(c, weight=1)

    frames = {}
    labels = {}

    position ={
        "top_left": (0,0), 
        "top_right": (0,2), 
        "bottom_left": (2, 0), 
        "bottom_right": (2, 2), 
        "center": (1, 1)
    }

    for name, (r, c) in position.items():
        frm = ttk.Frame(root, borderwidth=2, relief="solid")
        frm.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
        lbl = tk.Label(frm, bg="black")
        lbl.pack(expand=True, fill="both")
        frames[name] = frm
        labels[name] = lbl

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("couldn't open the webcam")
        return
    
    # for the face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    current_region = None
    photo_feedback = tk.StringVar(value="Say a region name to capture photo...")

    feedback_label = ttk.Label(root, textvariable=photo_feedback, foreground="blue", font=("Arial", 12, "bold"))
    feedback_label.grid(row=1, column=1)

    # for centering the images
    centered_frames = 0 
    center_threshold = 8
    capture_in_progress = False
    photo_taken = False
    centered_frames = 0

    countdown_label = ttk.Label(root, text="", foreground="red", font=("Arial", 36, "bold"))
    countdown_label.place(relx=0.5, rely=0.5, anchor="center")

    def start_countdown_and_capture(frame):
        nonlocal photo_taken, capture_in_progress
        capture_in_progress = True

        for i in [3, 2, 1, "Smile!"]:
            countdown_label.config(text=str(i))
            root.update()
            time.sleep(1)

        countdown_label.config(text="")
        filename = f"{current_region}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        photo_feedback.set(f"Image captured and saved as: {filename}")
        print(f"Saved: {filename}")

        capture_in_progress = False
        photo_taken = True

    def update_frame():
        nonlocal centered_frames, photo_taken, capture_in_progress

        ret, frame = cap.read()
        if not ret:
            root.after(30, update_frame)
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(60, 60))

        frame_h, frame_w = frame.shape[:2]
        frame_center_x, frame_center_y = frame_w // 2, frame_h // 2
        center_threshold = 50

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if (abs(face_center_x - frame_center_x) < center_threshold and
                    abs(face_center_y - frame_center_y) < center_threshold):
                centered_frames += 1
                photo_feedback.set("Face centered... Hold still!")
            else:
                centered_frames = 0
                photo_feedback.set("Please center your face.")
        else:
            centered_frames = 0
            photo_feedback.set("No face detected.")

        if current_region and not photo_taken and not capture_in_progress and centered_frames > 8:
            start_countdown_and_capture(frame)

        frame_resized = cv2.resize(frame, (320, 240))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        for name, lbl in labels.items():
            if name == current_region:
                lbl.imgtk = imgtk
                lbl.configure(image=imgtk)
            else:
                lbl.configure(image="", bg="black")

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

    def listen_for_commands():
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        nonlocal current_region, photo_taken, centered_frames

        while True:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for command...")
                audio = recognizer.listen(source)

            try:
                command = recognizer.recognize_google(audio).lower()
                print("Heard:", command)

                for region in position.keys():
                    if region.replace("_", " ") in command:
                        current_region = region
                        photo_taken = False  # <--- reset capture status for new command
                        centered_frames = 0
                        photo_feedback.set(f"Activated {region.replace('_', ' ').title()}. Please align your face in center.")
                        print(f"Activated {region}")
                        break
                else:
                    photo_feedback.set("Unknown command. Please say: top left, top right, etc.")

            except sr.UnknownValueError:
                print("Could not understand audio.")
                photo_feedback.set("Didn't catch that. Try again.")
            except sr.RequestError as e:
                print("Speech Recognition error:", e)
                photo_feedback.set("Speech recognition service error.")

    def on_close():
        cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    update_frame()
    threading.Thread(target=listen_for_commands, daemon=True).start()
    root.mainloop()
    
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

if __name__ == "__main__":
    main()
