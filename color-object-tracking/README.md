# Color Object Tracking with Direction Detection

Tracks a colored object via webcam and reports its position
(Left / Right / Front) and whether it's too close (Stop), based on
OpenCV contour detection + HSV color masking.

## Files

- `colorCalibrationForHSV_gui_v2.py` — GUI calibration tool (Tkinter).
  Use this FIRST to find the right HSV range for your object's color.
- `colorCalibrationForHSV.py` — simpler OpenCV-trackbar-only version of
  the calibrator (no GUI window, just trackbar sliders). Optional
  alternative if you don't want the Tkinter version.
- `main.py` — the actual tracking script. Reads a frame, masks the
  target color, finds the largest contour, draws a circle at its
  center, and prints/display the direction (Left/Right/Front/Stop).

## Setup

```bash
pip install opencv-python pillow imutils numpy
```

Tkinter ships with standard Python installs on Windows and Mac, so no
extra install is needed for it there. On some Linux distros you may
need `sudo apt install python3-tk`.

## Step 1 — Calibrate your color

```bash
python colorCalibrationForHSV_gui_v2.py
```

This opens a window matching the classic layout:

- **Top-left**: live grayscale/binary Mask preview (black background,
  white blob = detected object)
- **Top-right**: live camera feed
- **Hue / Saturation / Value** sections, each with a Low and High
  slider
- **Right-side buttons**:
  - `Print` — prints the current HSV Low/High values to the console
    and shows them in a popup
  - `Reds` / `Greens` / `Blues` — jump the sliders to a typical preset
    for that color, as a tuning starting point
  - `Open` — calibrate against a still image file instead of the live
    camera
  - `Screenshot` — saves the current camera frame + mask to
    `captures/` as PNGs
  - `Timer` — starts a 5-second countdown, then auto-takes a
    screenshot (useful for getting into position first)

Drag the Low/High sliders for each channel until **only your target
object appears white** in the Mask panel and everything else is
black. Click `Print` to get the exact values.

## Step 2 — Plug values into main.py

Open `main.py` and update:

```python
redLower = (95, 49, 100)     # <- your Low H, S, V from the calibrator
redUpper = (154, 255, 255)   # <- your High H, S, V from the calibrator
```

## Step 3 — Run the tracker

```bash
python main.py
```

Hold your calibrated object in front of the camera:

- Move it **left/right** in frame → prints/shows "Right" / "Left"
- Centered at normal distance → "Front"
- Bring it close to the camera (radius grows past 250) → "Stop"

Press `q` to quit.

## Notes

- If your webcam is external/USB rather than built-in, change
  `CAM_INDEX = 0` (calibrator) and `cv2.VideoCapture(0)` (main.py) to
  `1`.
- The Left/Right thresholds (`center[0] < 150`, `center[0] > 450`) and
  the Stop threshold (`radius > 250`) assume a 1000px-wide resized
  frame — adjust these numbers if you resize to a different width.
