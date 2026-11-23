import argparse
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ImagePoint:
    x: float
    y: float


# ---------------------------------------------------------------------------
# Click-to-detect: find the object under a click and return its top/bottom
# ---------------------------------------------------------------------------

def segment_object(image, seed, tolerance=12):
    """Grow a region from the clicked pixel, stopped by Canny edges.

    Returns a boolean mask of the object (or None if the result is implausible).
    Edges act as walls (4-connected fill cannot cross an 8-connected edge line),
    and colour similarity to the clicked pixel decides what else is included.
    """
    h, w = image.shape[:2]
    sx, sy = int(round(seed.x)), int(round(seed.y))
    if not (0 <= sx < w and 0 <= sy < h):
        return None

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    # floodFill needs a (h+2, w+2) mask where non-zero pixels are barriers.
    barrier = np.zeros((h + 2, w + 2), np.uint8)
    barrier[1:-1, 1:-1][edges > 0] = 1

    # Don't let the click itself land on a wall.
    y0, y1 = max(0, sy - 2), min(h, sy + 3)
    x0, x1 = max(0, sx - 2), min(w, sx + 3)
    barrier[1 + y0:1 + y1, 1 + x0:1 + x1] = 0

    tol = (tolerance,) * 3
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(blurred.copy(), barrier, (sx, sy), 0, tol, tol, flags)

    mask = (barrier[1:-1, 1:-1] == 255).astype(np.uint8)

    # Remove thin leaks / speckle, then keep the piece containing the click.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, labels = cv2.connectedComponents(mask, connectivity=4)
    label = labels[sy, sx]
    if label == 0:
        # Click was removed by the opening; use the nearest remaining blob.
        ys, xs = np.nonzero(labels)
        if len(xs) == 0:
            return None
        nearest = np.argmin((xs - sx) ** 2 + (ys - sy) ** 2)
        label = labels[ys[nearest], xs[nearest]]
    mask = labels == label

    area = int(mask.sum())
    if area < 50 or area > 0.6 * h * w:
        return None
    return mask


def endpoints_from_mask(mask):
    """Topmost and bottommost points of the object, x taken from the end rows."""
    ys, xs = np.nonzero(mask)
    top_y, bottom_y = int(ys.min()), int(ys.max())
    band = 2  # average x over the last few rows so one noisy pixel can't win

    top_xs = xs[ys <= top_y + band]
    bottom_xs = xs[ys >= bottom_y - band]
    return (
        ImagePoint(float(top_xs.mean()), float(top_y)),
        ImagePoint(float(bottom_xs.mean()), float(bottom_y)),
    )


def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def nearest_vertical_line(image, seed, max_distance=30):
    """Fallback: the near-vertical Hough line closest to the click."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    height = gray.shape[0]
    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=max(30, int(height * 0.10)),
        minLineLength=max(40, int(height * 0.10)),
        maxLineGap=max(10, int(height * 0.03)),
    )
    if lines is None:
        return None

    best = None
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, line)
        if abs(y2 - y1) < 2.5 * max(abs(x2 - x1), 1.0):
            continue  # not vertical enough
        dist = _point_to_segment_distance(seed.x, seed.y, x1, y1, x2, y2)
        if dist <= max_distance and (best is None or dist < best[0]):
            best = (dist, x1, y1, x2, y2)

    if best is None:
        return None
    _d, x1, y1, x2, y2 = best
    if y1 <= y2:
        return ImagePoint(x1, y1), ImagePoint(x2, y2)
    return ImagePoint(x2, y2), ImagePoint(x1, y1)


def detect_object_endpoints(image, seed, tolerance=12):
    """Return (top, bottom, mask_or_None) for the object at `seed`, or None."""
    mask = segment_object(image, seed, tolerance)
    if mask is not None:
        top, bottom = endpoints_from_mask(mask)
        return top, bottom, mask

    line = nearest_vertical_line(image, seed)
    if line is not None:
        return line[0], line[1], None
    return None


# ---------------------------------------------------------------------------
# Interactive window
# ---------------------------------------------------------------------------

def _draw_overlay(image, mask, top, bottom, seed, status):
    display = image.copy()

    if mask is not None:
        tint = display.copy()
        tint[mask] = (0, 200, 255)
        display = cv2.addWeighted(tint, 0.4, display, 0.6, 0)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, (0, 255, 255), 1)

    if seed is not None:
        cv2.drawMarker(display, (round(seed.x), round(seed.y)), (255, 128, 0),
                       cv2.MARKER_TILTED_CROSS, 14, 1)
    if top is not None:
        cv2.circle(display, (round(top.x), round(top.y)), 6, (0, 0, 255), 2)
        cv2.putText(display, f"top ({top.x:.0f}, {top.y:.0f})",
                    (round(top.x) + 10, round(top.y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    if bottom is not None:
        cv2.circle(display, (round(bottom.x), round(bottom.y)), 6, (0, 255, 0), 2)
        cv2.putText(display, f"bottom ({bottom.x:.0f}, {bottom.y:.0f})",
                    (round(bottom.x) + 10, round(bottom.y) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Status bar
    cv2.rectangle(display, (0, 0), (display.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(display, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return display


def pick_object_endpoints(image, message, window_name):
    """Click the object; OpenCV finds its top/bottom. Returns (top, bottom).

    Controls
      Left click    pick the object (re-click to try again)
      + / -         loosen / tighten the colour tolerance and re-detect
      M             manual mode: click the top, then the bottom
      Enter         accept
      Esc           cancel
    """
    state = {
        "seed": None, "tol": 12, "top": None, "bottom": None, "mask": None,
        "manual": False, "manual_clicks": [], "note": "Click on the object.",
    }

    def detect():
        result = detect_object_endpoints(image, state["seed"], state["tol"])
        if result is None:
            state.update(top=None, bottom=None, mask=None,
                         note="No object found - click again, press +, or M for manual.")
        else:
            state["top"], state["bottom"], state["mask"] = result
            kind = "region" if result[2] is not None else "edge line"
            state["note"] = f"Found ({kind}), tol={state['tol']}. Enter=accept  +/-=tolerance  M=manual"

    def refresh():
        cv2.imshow(window_name, _draw_overlay(
            image, state["mask"], state["top"], state["bottom"],
            state["seed"], state["note"]))

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["manual"]:
            state["manual_clicks"].append(ImagePoint(float(x), float(y)))
            if len(state["manual_clicks"]) == 1:
                state["top"], state["bottom"], state["mask"] = state["manual_clicks"][0], None, None
                state["note"] = "Manual: now click the BOTTOM point."
            else:
                state["bottom"] = state["manual_clicks"][1]
                state["manual"] = False
                state["manual_clicks"] = []
                state["note"] = "Manual points set. Enter=accept, or click the object to re-detect."
        else:
            state["seed"] = ImagePoint(float(x), float(y))
            detect()
        refresh()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print(message)
    print("Click the object. Enter = accept, +/- = tolerance, M = manual, Esc = cancel.")
    refresh()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and state["top"] is not None and state["bottom"] is not None:
            cv2.destroyWindow(window_name)
            return state["top"], state["bottom"]
        if key == 27:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt
        if key in (ord("+"), ord("=")) and state["seed"] is not None:
            state["tol"] = min(80, state["tol"] + 4)
            detect()
            refresh()
        elif key in (ord("-"), ord("_")) and state["seed"] is not None:
            state["tol"] = max(2, state["tol"] - 4)
            detect()
            refresh()
        elif key in (ord("m"), ord("M")):
            state["manual"] = True
            state["manual_clicks"] = []
            state["note"] = "Manual: click the TOP point."
            refresh()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def calculate_measurement(
    image_1_bottom,
    image_2_bottom,
    image_2_top,
    baseline,
):
    disparity = abs(image_1_bottom.x - image_2_bottom.x)
    if disparity < 1.0:
        raise ValueError("The bottom-point displacement is too small to calculate distance.")

    ruler = baseline / disparity
    height = ruler * (abs(image_2_bottom.y - image_2_top.y))

    return height, disparity


def capture_image(camera, label):
    print(f"\n{label}: press Space to capture, or Q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Unable to read a frame from the webcam.")

        cv2.imshow("Remote Sensing Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt
        if key == 32:
            print(f"Captured {label}.")
            return frame.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=float, required=True, help="Horizontal movement in feet")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    if args.baseline <= 0:
        raise SystemExit("Baseline must be positive.")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1.")

    try:
        print("Keep the laptop level and pointed in the same direction for all images.")
        print("Keep the camera height unchanged and keep the entire object visible if possible.")

        image_1 = capture_image(camera, "Image 1: original position")
        image_2 = capture_image(camera, "Image 2: move horizontally by the baseline")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    image_1_top, image_1_bottom = pick_object_endpoints(
        image_1, "Image 1: click on the object.", "Image 1 - click the object")
    image_2_top, image_2_bottom = pick_object_endpoints(
        image_2, "Image 2: click on the same object.", "Image 2 - click the object")

    try:
        height, disparity = calculate_measurement(
            image_1_bottom,
            image_2_bottom,
            image_2_top,
            args.baseline,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print("\nEndpoint pixel coordinates (x, y)")
    print(f"Image 1 top:    ({image_1_top.x:.1f}, {image_1_top.y:.1f})")
    print(f"Image 1 bottom: ({image_1_bottom.x:.1f}, {image_1_bottom.y:.1f})")
    print(f"Image 2 top:    ({image_2_top.x:.1f}, {image_2_top.y:.1f})")
    print(f"Image 2 bottom: ({image_2_bottom.x:.1f}, {image_2_bottom.y:.1f})")

    print("\nMeasurement result")
    print(f"Bottom-point disparity: {disparity:.2f} pixels")
    print(f"Object height:          {height:.3f} ft")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nMeasurement cancelled.")
