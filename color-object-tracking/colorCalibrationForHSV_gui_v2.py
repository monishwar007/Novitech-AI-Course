"""
colorCalibrationForHSV.py  (GUI version - matches original tutorial layout)
----------------------------------------------------------------------------
Layout:
    [ Mask (grayscale B&W) ]   [ Live Camera Feed ]
    ------------------------------------------------
    Hue
        Low  ----slider----
        High ----slider----
    Saturation
        Low  ----slider----
        High ----slider----
    Value
        Low  ----slider----
        High ----slider----
                                    [Print]
                                    [Reds]
                                    [Greens]
                                    [Blues]
                                    [Open]
                                    [Screenshot]
                                    [Timer]

Buttons:
    Print       -> prints current HSV Low/High values to the console
    Reds        -> jumps sliders to a typical RED preset
    Greens      -> jumps sliders to a typical GREEN preset
    Blues       -> jumps sliders to a typical BLUE preset
    Open        -> opens an image file from disk and calibrates on it
                   instead of the live camera (falls back to camera if
                   cancelled)
    Screenshot  -> saves the current camera frame + mask to disk as PNGs
    Timer       -> starts a 5-second countdown, then auto-clicks Screenshot
                   (handy for getting into position before it captures)

The Mask panel is the grayscale/binary result of cv2.inRange() -- pure
black & white, exactly like in the original tool -- so you can see
precisely what the mask will look like in main.py.

Install:
    pip install opencv-python pillow numpy

Run:
    python colorCalibrationForHSV.py
"""

import os
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

CAM_INDEX = 0            # change to 1 for an external/USB webcam
PANEL_W, PANEL_H = 380, 280   # size of each video/mask panel


class HSVCalibratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sliders")
        self.root.resizable(False, False)

        self.camera = cv2.VideoCapture(CAM_INDEX)
        if not self.camera.isOpened():
            messagebox.showerror("Camera Error",
                                  f"Could not open camera index {CAM_INDEX}.")
            self.root.destroy()
            return

        # If not None, calibrate against this still image instead of the
        # live camera feed (set via the "Open" button).
        self.static_image = None

        # Timer state
        self.timer_remaining = 0
        self.timer_running = False

        # Tkinter variables backing each slider
        self.h_low = tk.IntVar(value=0)
        self.h_high = tk.IntVar(value=179)
        self.s_low = tk.IntVar(value=0)
        self.s_high = tk.IntVar(value=255)
        self.v_low = tk.IntVar(value=100)
        self.v_high = tk.IntVar(value=255)

        os.makedirs("captures", exist_ok=True)

        self._build_layout()
        self._update_frame()

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0)

        # ---- Top: Mask (left) + Camera feed (right) ---- #
        video_row = ttk.Frame(outer)
        video_row.grid(row=0, column=0, columnspan=2, pady=(0, 5))

        self.mask_label = tk.Label(video_row, bg="black")
        self.mask_label.grid(row=0, column=0, padx=(0, 5))

        self.frame_label = tk.Label(video_row)
        self.frame_label.grid(row=0, column=1, padx=(5, 0))

        # ---- Slider sections: Hue / Saturation / Value ---- #
        sliders_col = ttk.Frame(outer)
        sliders_col.grid(row=1, column=0, sticky="nw", pady=(5, 0))

        self._build_channel_section(sliders_col, "Hue", "#c0392b",
                                     self.h_low, self.h_high, 179, row=0)
        self._build_channel_section(sliders_col, "Saturation", "#27ae60",
                                     self.s_low, self.s_high, 255, row=1)
        self._build_channel_section(sliders_col, "Value", "#2980b9",
                                     self.v_low, self.v_high, 255, row=2)

        # ---- Right-side button column ---- #
        btn_col = ttk.Frame(outer)
        btn_col.grid(row=1, column=1, sticky="ne", padx=(20, 0))

        ttk.Button(btn_col, text="Print", width=12,
                   command=self._print_values).pack(pady=4)
        ttk.Button(btn_col, text="Reds", width=12,
                   command=lambda: self._apply_preset(0, 10, 120, 255, 70, 255)
                   ).pack(pady=4)
        ttk.Button(btn_col, text="Greens", width=12,
                   command=lambda: self._apply_preset(40, 80, 80, 255, 70, 255)
                   ).pack(pady=4)
        ttk.Button(btn_col, text="Blues", width=12,
                   command=lambda: self._apply_preset(95, 130, 80, 255, 70, 255)
                   ).pack(pady=4)
        ttk.Button(btn_col, text="Open", width=12,
                   command=self._open_image).pack(pady=4)
        ttk.Button(btn_col, text="Screenshot", width=12,
                   command=self._take_screenshot).pack(pady=4)
        self.timer_btn = ttk.Button(btn_col, text="Timer", width=12,
                                     command=self._start_timer)
        self.timer_btn.pack(pady=4)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_channel_section(self, parent, title, color, low_var, high_var,
                                max_val, row):
        section = ttk.Frame(parent, padding=(0, 6))
        section.grid(row=row, column=0, sticky="w")

        ttk.Label(section, text=title, foreground=color,
                  font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")

        # --- Low --- #
        low_value_lbl = ttk.Label(section, text=str(low_var.get()))
        ttk.Label(section, text="Low").grid(row=1, column=0, sticky="w")
        low_value_lbl.grid(row=1, column=1, sticky="w", padx=(6, 0))
        low_scale = ttk.Scale(section, from_=0, to=max_val, orient="horizontal",
                               length=480, variable=low_var,
                               command=lambda _=None: self._on_slider_change())
        low_scale.grid(row=2, column=0, columnspan=2, sticky="w")

        # --- High --- #
        high_value_lbl = ttk.Label(section, text=str(high_var.get()))
        ttk.Label(section, text="High").grid(row=3, column=0, sticky="w")
        high_value_lbl.grid(row=3, column=1, sticky="w", padx=(6, 0))
        high_scale = ttk.Scale(section, from_=0, to=max_val, orient="horizontal",
                                length=480, variable=high_var,
                                command=lambda _=None: self._on_slider_change())
        high_scale.grid(row=4, column=0, columnspan=2, sticky="w")

        if not hasattr(self, "_value_labels"):
            self._value_labels = []
        self._value_labels.append((low_var, low_value_lbl))
        self._value_labels.append((high_var, high_value_lbl))

    def _on_slider_change(self):
        for var in (self.h_low, self.h_high, self.s_low,
                    self.s_high, self.v_low, self.v_high):
            var.set(int(round(var.get())))
        for var, lbl in self._value_labels:
            lbl.configure(text=str(int(var.get())))

    def _apply_preset(self, hl, hh, sl, sh, vl, vh):
        self.h_low.set(hl); self.h_high.set(hh)
        self.s_low.set(sl); self.s_high.set(sh)
        self.v_low.set(vl); self.v_high.set(vh)
        self._on_slider_change()

    # ------------------------------------------------------------------ #
    # Core video / mask loop
    # ------------------------------------------------------------------ #
    def _get_source_frame(self):
        """Returns a BGR frame either from the static opened image or
        the live camera."""
        if self.static_image is not None:
            return self.static_image.copy()
        grabbed, frame = self.camera.read()
        if not grabbed:
            return None
        return cv2.flip(frame, 1)

    def _update_frame(self):
        frame = self._get_source_frame()

        if frame is not None:
            frame = cv2.resize(frame, (PANEL_W, PANEL_H))

            # Convert to HSV, then mask -> the mask itself IS a grayscale
            # (single channel, black/white) image, matching the original tool.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([self.h_low.get(), self.s_low.get(), self.v_low.get()])
            upper = np.array([self.h_high.get(), self.s_high.get(), self.v_high.get()])
            mask = cv2.inRange(hsv, lower, upper)  # single-channel grayscale mask

            self._last_frame = frame
            self._last_mask = mask

            # Camera panel (BGR -> RGB for Tkinter)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.frame_label.configure(image=frame_img)
            self.frame_label.image = frame_img

            # Mask panel — keep it strictly grayscale (mode "L"), not RGB,
            # so it renders as true black & white like the original.
            mask_img = ImageTk.PhotoImage(Image.fromarray(mask, mode="L"))
            self.mask_label.configure(image=mask_img)
            self.mask_label.image = mask_img

        # Timer countdown tick
        if self.timer_running:
            now = time.time()
            remaining = int(self.timer_target - now) + 1
            if remaining <= 0:
                self.timer_running = False
                self.timer_btn.configure(text="Timer")
                self._take_screenshot()
            else:
                self.timer_btn.configure(text=f"{remaining}s...")

        self.root.after(15, self._update_frame)

    # ------------------------------------------------------------------ #
    # Button actions
    # ------------------------------------------------------------------ #
    def _print_values(self):
        text = (
            f"redLower = ({self.h_low.get()}, {self.s_low.get()}, {self.v_low.get()})\n"
            f"redUpper = ({self.h_high.get()}, {self.s_high.get()}, {self.v_high.get()})"
        )
        print(text)  # matches original tool's "Print" button behaviour
        messagebox.showinfo("HSV Values", text)

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open an image to calibrate against",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.static_image = img
            else:
                messagebox.showerror("Error", "Could not read that image file.")
        else:
            # user cancelled -> go back to live camera
            self.static_image = None

    def _take_screenshot(self):
        if not hasattr(self, "_last_frame"):
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        frame_path = os.path.join("captures", f"frame_{ts}.png")
        mask_path = os.path.join("captures", f"mask_{ts}.png")
        cv2.imwrite(frame_path, self._last_frame)
        cv2.imwrite(mask_path, self._last_mask)
        print(f"Saved {frame_path} and {mask_path}")
        messagebox.showinfo("Screenshot Saved",
                             f"Saved:\n{frame_path}\n{mask_path}")

    def _start_timer(self, seconds=5):
        self.timer_running = True
        self.timer_target = time.time() + seconds

    def _on_close(self):
        self.camera.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HSVCalibratorApp(root)
    root.mainloop()
