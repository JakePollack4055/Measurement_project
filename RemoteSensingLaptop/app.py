import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class ImagePoint:
    x: float
    y: float


def choose_point(image, message, window_name):
    """Display an image and return one manually selected pixel point."""
    selected = []
    display = image.copy()

    def on_mouse(event, x, y, _flags, _param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.clear()
            selected.append(ImagePoint(float(x), float(y)))
            display = image.copy()
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
    image_1_bottom = choose_point(
        image_1,
        "Select the bottom of the object in Image 1.",
        "Select Image 1 bottom",
    )
    image_2_bottom = choose_point(
        image_2,
        "Select the same bottom point in Image 2.",
        "Select Image 2 bottom",
    )
    image_2_top = choose_point(
        image_2,
        "Select the top of the object in Image 2.",
        "Select Image 2 top",
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\nMeasurement cancelled.")

