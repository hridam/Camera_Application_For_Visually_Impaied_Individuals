import tkinter as tk
from tkinter import ttk

def main():
    root = tk.Tk()
    root.title("Five Regions")
    root.geometry("800x500")

    # main_frame = ttk.Frame(root, padding=5)
    # main_frame.grid(row=0, column=0, sticky="nsew")

    # root.rowconfigure((0,1), weight=1)
    # root.columnconfigure((0,1), weight=1)

    for r in range(3):
        root.rowconfigure(r, weight=1)
    for c in range(3):
        root.columnconfigure(c, weight=1)

    
    # main_frame.rowconfigure((0, 1), weight=1)
    # main_frame.columnconfigure((0,1), weight=1)

    top_left = ttk.Frame(root, borderwidth=2, relief="solid")
    top_left.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
    ttk.Label(top_left, text="Top Left",).pack(expand=True)

    top_right = ttk.Frame(root, borderwidth=2, relief="solid")
    top_right.grid(row=0, column=2, sticky="nsew", padx=0, pady=0)
    ttk.Label(top_right, text="Top Right",).pack(expand=True)

    bottom_left = ttk.Frame(root, borderwidth=2, relief="solid")
    bottom_left.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)
    ttk.Label(bottom_left, text="Top Left",).pack(expand=True)

    bottom_right = ttk.Frame(root, borderwidth=2, relief="solid")
    bottom_right.grid(row=2, column=2, sticky="nsew", padx=0, pady=0)
    ttk.Label(bottom_right, text="Top Left",).pack(expand=True)

    # center = ttk.Frame(root, borderwidth=2, relief="solid")
    # center.place(relx=0.5, rely=0.5, anchor="center", width=500, height=400)
    # ttk.Label(center, text="Top Left",).pack(expand=True)
    center = ttk.Frame(root, borderwidth=2, relief="solid")
    center.grid(row=1, column=1, sticky="nsew")
    ttk.Label(center, text="Center").pack(expand=True)

    # ttk.Label(top_left, text="Top Left",).pack(expand=True)
    # ttk.Label(top_right, text="Top Right",).pack(expand=True)
    # ttk.Label(bottom_left, text="Top Left",).pack(expand=True)
    # ttk.Label(bottom_right, text="Top Left",).pack(expand=True)
    # ttk.Label(center, text="Top Left",).pack(expand=True)

    root.mainloop()

if __name__ == "__main__":
    main()