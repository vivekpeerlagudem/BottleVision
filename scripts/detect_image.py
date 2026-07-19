"""BottleVision single-image detection tool (Milestone M2).

Loads a pretrained YOLO model once, runs it on one image given on the command
line, prints every detection, and saves an annotated copy.

Run it with::

    python scripts/detect_image.py path/to/image.jpg
    python scripts/detect_image.py path/to/image.jpg -o out.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bottlevision.config import ConfigError, load_config
from bottlevision.detection import Detection
from bottlevision.detector import Detector
from bottlevision.utils.logging import configure_logging, get_logger

# Drawing colours are BGR (OpenCV's channel order), not RGB.
_BOX_COLOR = (0, 255, 0)  # green


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a pretrained YOLO model on a single image."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Where to save the annotated image "
        "(default: alongside the input as '<name>_detected<ext>').",
    )
    return parser.parse_args()


def _draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw boxes and labels onto a copy of the image.

    NOTE: This is a temporary helper local to the M2 tool. In Milestone M5 it is
    replaced by the reusable ``annotator.py`` module, so drawing logic is not
    duplicated across the project.
    """
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _BOX_COLOR,
            1,
            cv2.LINE_AA,
        )
    return annotated


def main() -> None:
    """Load a model, detect on one image, print results, and save the output."""
    configure_logging()
    log = get_logger("bottlevision.detect_image")
    args = _parse_args()

    if not args.image.is_file():
        log.error("Input image not found: %s", args.image)
        raise SystemExit(1)

    image = cv2.imread(str(args.image))
    if image is None:
        log.error("Could not read image (unsupported or corrupt file): %s", args.image)
        raise SystemExit(1)

    try:
        config = load_config()
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    detector = Detector(config.detector)
    detections = detector.detect(image)

    if not detections:
        log.info("No objects detected at or above the confidence threshold.")
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox
        log.info(
            "[%d] %-15s conf=%.2f  box=(%d, %d, %d, %d)",
            i,
            det.class_name,
            det.confidence,
            x1,
            y1,
            x2,
            y2,
        )

    annotated = _draw_detections(image, detections)
    output_path = args.output or args.image.with_name(
        f"{args.image.stem}_detected{args.image.suffix}"
    )
    cv2.imwrite(str(output_path), annotated)
    log.info("Saved annotated image to: %s", output_path)


if __name__ == "__main__":
    main()
