"""Run confidence-filtered YOLO traffic-sign detection on a local webcam."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GTSDB webcam inference.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "task8_traffic_sign_recognition/outputs/models/"
            "gtsdb_yolov8n_best.pt"
        ),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")

    previous = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            result = model.predict(
                frame,
                imgsz=args.image_size,
                conf=args.confidence,
                verbose=False,
            )[0]
            display = result.plot()
            now = time.perf_counter()
            fps = 1 / max(now - previous, 1e-9)
            previous = now
            cv2.putText(
                display,
                f"FPS: {fps:.1f}  conf >= {args.confidence:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("GTSDB YOLO Traffic Sign Detection", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
