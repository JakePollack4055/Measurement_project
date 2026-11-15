import json
from pathlib import Path

import cv2
import numpy as np


CHECKERBOARD = (9, 6)
SQUARE_SIZE_INCHES = 1.0


def main():
    object_points = []
    image_points = []

    object_pattern = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    object_pattern[:, :2] = np.mgrid[
        0:CHECKERBOARD[0], 0:CHECKERBOARD[1]
    ].T.reshape(-1, 2)
    object_pattern *= SQUARE_SIZE_INCHES

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam.")

    print("Show a 9 by 6 checkerboard to the webcam.")
    print("Press Space to capture a calibration view; press Q when finished.")

    image_size = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # The SB detector is more reliable than the classic detector for
            # webcams, perspective, and uneven lighting. Fall back when an
            # older OpenCV build does not provide it.
            if hasattr(cv2, "findChessboardCornersSB"):
                found, corners = cv2.findChessboardCornersSB(
                    gray,
                    CHECKERBOARD,
                    flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
                )
            else:
                found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, found)
                cv2.putText(preview, "Pattern found", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(preview, "Show checkerboard", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Camera calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and found:
                refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                object_points.append(object_pattern.copy())
                image_points.append(refined)
                print(f"Captured calibration view {len(image_points)}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if len(image_points) < 5 or image_size is None:
        raise SystemExit("Capture at least five valid checkerboard views.")

    _error, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    calibration = {
        "focal_length_x_pixels": float(camera_matrix[0, 0]),
        "focal_length_y_pixels": float(camera_matrix[1, 1]),
        "principal_point_x_pixels": float(camera_matrix[0, 2]),
        "principal_point_y_pixels": float(camera_matrix[1, 2]),
        "distortion": distortion.flatten().tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1],
    }

    Path("camera_calibration.json").write_text(json.dumps(calibration, indent=2))
    print("Saved camera_calibration.json")
    print(f"Use focal length: {(calibration['focal_length_x_pixels'] + calibration['focal_length_y_pixels']) / 2:.2f}")


if __name__ == "__main__":
    main()

