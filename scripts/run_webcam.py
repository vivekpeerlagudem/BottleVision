"""BottleVision entry point.

The thin script you actually run. Its job is to: configure logging, load the
configuration, build the pipeline stations, and start the loop. All real logic
lives in the ``bottlevision`` package -- this script only wires things together.

Run it with::

    python scripts/run_webcam.py
"""

from __future__ import annotations

from bottlevision.annotator import Annotator
from bottlevision.camera import Camera, CameraError
from bottlevision.config import ConfigError, load_config
from bottlevision.detector import Detector
from bottlevision.display import Display
from bottlevision.pipeline import Pipeline
from bottlevision.postprocess import Postprocessor
from bottlevision.tracker import Tracker
from bottlevision.utils.logging import configure_logging, get_logger


def main() -> None:
    """Configure, wire, and run the BottleVision webcam pipeline."""
    configure_logging()
    log = get_logger("bottlevision.run")

    try:
        config = load_config()
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    # Build the six stations. The detector loads its model here, once, before
    # the loop starts -- so per-frame cost is only the forward pass. YOLO is
    # queried at the bottle-equivalent threshold so class-specific policy in
    # the postprocessor can see borderline detections.
    camera = Camera(config.camera)
    display = Display(config.display)
    detector = Detector(
        config.detector,
        min_query_confidence=config.filter.bottle_confidence_threshold,
    )
    postprocessor = Postprocessor(config.filter)
    tracker = Tracker(config.tracker)
    annotator = Annotator()
    pipeline = Pipeline(
        camera, display, detector, postprocessor, tracker, annotator
    )

    try:
        pipeline.run()
    except CameraError as exc:
        log.error("Camera error: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")


if __name__ == "__main__":
    main()
