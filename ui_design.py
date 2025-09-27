import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

def main():
    root = tk.Tk()
    root.title("Five Regions")
    root.geometry("800x500")

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
    lables = {}

    position ={
        "top_left": (0,0), 
        "top_rigth": (0,2), 
        "bottom_left": (2, 0), 
        "bottom_rigth": (2, 2), 
        "center": (1, 1)
    }

    for name, (r, c) in position.items():
        frm = ttk.Frame(root, borderwidth=2, relief="solid")
        frm.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
        lbl = tk.Label(frm)
        lbl.pack(expand=True, fill="both")
        frames[name] = frm
        lables[name] = lbl

    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 240))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            for lbl in lables.values():
                lbl.imgtk = imgtk
                lbl.configure(image=imgtk)

        root.after(30, update_frame)
    
    update_frame()

    def on_close():
        cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
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
