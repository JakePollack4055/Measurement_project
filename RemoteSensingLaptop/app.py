import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class ImagePoint:
    x: float
    y: float


def detect_vertical_edge(image):
    """Return a likely vertical object edge as (top_point, bottom_point)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height, width = gray.shape[:2]
    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 180.0,
        threshold=max(30, int(height * 0.10)),
        minLineLength=max(40, int(height * 0.20)),
        maxLineGap=max(10, int(height * 0.03)),
    )

    candidates = []
    if lines is not None:
        # OpenCV can return Hough lines with different nesting shapes across
        # versions. Normalize them to one row per [x1, y1, x2, y2].
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(float, line)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy < 2.5 * max(dx, 1.0):
                continue

            length = math.hypot(x2 - x1, y2 - y1)
            center_x = (x1 + x2) / 2.0
            center_penalty = abs(center_x - width / 2.0) / width
            score = length * (1.0 - 0.7 * center_penalty)
            candidates.append((score, x1, y1, x2, y2))

    if not candidates:
        return None

    _score, x1, y1, x2, y2 = max(candidates)
    if y1 <= y2:
        return ImagePoint(x1, y1), ImagePoint(x2, y2)
    return ImagePoint(x2, y2), ImagePoint(x1, y1)


def choose_point(image, message, window_name, suggested=None, line=None):
    """Display an image and return one manually selected pixel point."""
    selected = []
    display = image.copy()

    if line is not None:
        top, bottom = line
        cv2.line(
            display,
            (round(top.x), round(top.y)),
            (round(bottom.x), round(bottom.y)),
            (0, 255, 255),
            2,
        )

    if suggested is not None:
        cv2.drawMarker(
            display,
            (round(suggested.x), round(suggested.y)),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            24,
            2,
        )

    def on_mouse(event, x, y, _flags, _param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.clear()
            selected.append(ImagePoint(float(x), float(y)))
            display = image.copy()
            if line is not None:
                top, bottom = line
                cv2.line(
                    display,
                    (round(top.x), round(top.y)),
                    (round(bottom.x), round(bottom.y)),
                    (0, 255, 255),
                    2,
                )
            cv2.drawMarker(
                display,
                (x, y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                24,
                2,
            )
            cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)
    print(message)
    if suggested is not None:
        print("A green marker is the automatic suggestion. Press Enter to accept it, or click to correct it.")
    else:
        print("Click the requested point, then press Enter while the image window is active.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and selected:
            cv2.destroyWindow(window_name)
            return selected[0]
        if key == 27:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt


def angle_from_pixel(y, image_height, focal_length):
    principal_y = image_height / 2.0
    return math.atan2(principal_y - y, focal_length)


def calculate_measurement(
    image_1_bottom,
    image_2_bottom,
    image_2_top,
    image_height,
    baseline,
    focal_length,
):
    disparity = abs(image_1_bottom.x - image_2_bottom.x)
    if disparity < 1.0:
        raise ValueError("The bottom-point displacement is too small to calculate distance.")

    distance = focal_length * baseline / disparity

    bottom_angle = angle_from_pixel(
        image_2_bottom.y, image_height, focal_length
    )
    top_angle = angle_from_pixel(
        image_2_top.y, image_height, focal_length
    )
    height = distance * (math.tan(top_angle) - math.tan(bottom_angle))

    return distance, height, disparity, bottom_angle, top_angle


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
    parser.add_argument("--focal-length", type=float, required=True, help="Calibrated focal length in pixels")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    if args.baseline <= 0 or args.focal_length <= 0:
        raise SystemExit("Baseline and focal length must be positive.")

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

    h, _w = image_2.shape[:2]
    detected_1 = detect_vertical_edge(image_1)
    detected_2 = detect_vertical_edge(image_2)

    if detected_1 is None:
        print("OpenCV could not find a vertical edge in Image 1; manual selection is required.")
    if detected_2 is None:
        print("OpenCV could not find a vertical edge in Image 2; manual selection is required.")

    image_1_bottom = choose_point(
        image_1,
        "Select the bottom of the object in Image 1.",
        "Select Image 1 bottom",
        suggested=detected_1[1] if detected_1 else None,
        line=detected_1,
    )
    image_2_bottom = choose_point(
        image_2,
        "Select the same bottom point in Image 2.",
        "Select Image 2 bottom",
        suggested=detected_2[1] if detected_2 else None,
        line=detected_2,
    )
    image_2_top = choose_point(
        image_2,
        "Select the top of the object in Image 2.",
        "Select Image 2 top",
        suggested=detected_2[0] if detected_2 else None,
        line=detected_2,
    )

    try:
        distance, height, disparity, bottom_angle, top_angle = calculate_measurement(
            image_1_bottom,
            image_2_bottom,
            image_2_top,
            h,
            args.baseline,
            args.focal_length,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print("\nMeasurement result")
    print(f"Bottom-point disparity: {disparity:.2f} pixels")
    print(f"Distance to object:     {distance:.3f} ft")
    print(f"Bottom angle:           {math.degrees(bottom_angle):.3f} degrees")
    print(f"Top angle:              {math.degrees(top_angle):.3f} degrees")
    print(f"Object height:          {height:.3f} ft")
    print([image_1_bottom, image_1_top, image_2_top])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nMeasurement cancelled.")

