import os
import threading
import tkinter as tk
from tkinter import messagebox

import cv2
from deepface import DeepFace


def login_screen():
    """Shows a login screen to get username and load the reference image."""
    root = tk.Tk()
    root.title("Login")
    root.geometry("350x150")

    username_val = tk.StringVar()

    def on_login(event=None):
        if username_val.get():
            root.destroy()

    def on_closing():
        username_val.set("")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) / 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) / 2
    root.geometry("+%d+%d" % (x, y))

    tk.Label(root, text="Enter your username:").pack(pady=10, padx=10)
    entry = tk.Entry(root, textvariable=username_val, width=40)
    entry.pack(pady=10, padx=10)
    entry.focus_set()
    entry.bind("<Return>", on_login)

    tk.Button(root, text="Login", command=on_login).pack(pady=10)

    root.mainloop()

    username = username_val.get()
    if not username:
        print("No username entered. Exiting.")
        return None

    reference_img_path = f"{username}_ref.jpg"
    if not os.path.exists(reference_img_path):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Reference image not found: {reference_img_path}")
        root.destroy()
        return None

    return cv2.imread(reference_img_path)


def main():
    """Main function to run the face recognition app."""
    reference_img = login_screen()
    if reference_img is None:
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_match = False

    def check_face(frame):
        nonlocal face_match
        try:
            if DeepFace.verify(frame, reference_img.copy())['verified']:
                face_match = True
            else:
                face_match = False
        except ValueError:
            face_match = False

    counter = 0
    while True:
        ret, frame = cap.read()

        if ret:
            if counter % 30 == 0:
                try:
                    threading.Thread(target=check_face, args=(frame.copy(),)).start()
                except ValueError:
                    pass
            counter += 1
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

            cv2.imshow('video', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
