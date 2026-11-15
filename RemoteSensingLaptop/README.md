# Remote Sensing Laptop Prototype

This Windows prototype uses a laptop webcam, OpenCV, and a calibrated pinhole-camera model.

The laptop must remain level and pointed in the same direction for both images. Only the horizontal position changes between Images 1 and 2. The complete vertical object must remain visible in Image 2.

## Install

```powershell
py -m pip install -r requirements.txt
```

## Run

```powershell
py app.py --baseline 1.0 --focal-length 1200
```

Press:

- `1` to capture Image 1
- `2` to capture Image 2 after moving horizontally by the entered baseline
- `r` to reset the measurement
- `q` to quit

After capture, the program asks you to click:

1. The object bottom in Image 1
2. The object bottom in Image 2
3. The object top in Image 2

The clicks are temporary image annotations; the program does not interact with the physical object.

## Camera calibration

The focal length must be in pixels. The `--focal-length 1200` value is only a starting estimate. For better results, run the checkerboard calibration script and use the saved focal length.

```powershell
py calibrate_camera.py
```

