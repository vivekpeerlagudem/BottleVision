"""BottleVision entry point.

The thin script you actually run. Its job is to: configure logging, load the
configuration, build the pipeline stations, and start the loop. All real logic
lives in the ``bottlevision`` package -- this script only wires things together.

Run it with::

    python scripts/run_webcam.py
    python scripts/run_webcam.py --debug   # verbose per-frame tracker diagnostics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bottlevision.annotator import Annotator
from bottlevision.camera import Camera, CameraError
from bottlevision.config import ConfigError, load_config
from bottlevision.detector import Detector
from bottlevision.diagnostics import PipelineTracer
from bottlevision.display import Display
from bottlevision.pipeline import Pipeline
from bottlevision.postprocess import Postprocessor
from bottlevision.tracker import Tracker
from bottlevision.utils.logging import configure_logging, get_logger


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the BottleVision webcam pipeline."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose per-frame tracker diagnostics (timing, IoU matrix, "
        "matches, spawned/retired tracks).",
    )
    parser.add_argument(
        "--trace-dropouts",
        action="store_true",
        help="Investigate why frames produce no bottle: query YOLO at a low "
        "confidence to reveal sub-threshold detections, log every filtering "
        "decision, and save frames containing no bottle to disk.",
    )
    parser.add_argument(
        "--trace-confidence",
        type=float,
        default=0.05,
        help="Confidence used to query YOLO when --trace-dropouts is set "
        "(default: 0.05). The configured threshold is still applied before "
        "anything reaches the pipeline.",
    )
    parser.add_argument(
        "--dropout-dir",
        type=Path,
        default=Path("diagnostics/dropout_frames"),
        help="Where --trace-dropouts writes frames (default: "
        "diagnostics/dropout_frames).",
    )
    return parser.parse_args()


def main() -> None:
    """Configure, wire, and run the BottleVision webcam pipeline."""
    args = _parse_args()
    configure_logging(logging.DEBUG if args.debug else logging.INFO)
    log = get_logger("bottlevision.run")

    try:
        config = load_config()
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    # Build the four stations. The detector loads its model here, once, before
    # the loop starts -- so per-frame cost is only the forward pass.
    tracing = args.trace_dropouts
    if tracing:
        log.info(
            "Dropout tracing ON: querying YOLO at conf=%.2f (configured "
            "threshold %.2f still applies before the pipeline).",
            args.trace_confidence,
            config.detector.confidence_threshold,
        )

    camera = Camera(config.camera)
    display = Display(config.display)
    detector = Detector(
        config.detector,
        diagnostic_confidence=args.trace_confidence if tracing else None,
        # Query YOLO at the lower of the two thresholds so bottle-equivalent
        # detections below 0.50 (typically horizontal bottles labelled 'vase')
        # can reach the postprocessor and pass its class-specific rule. Non
        # bottle-equivalent classes are still dropped by the class filter.
        min_query_confidence=config.filter.bottle_confidence_threshold,
    )
    postprocessor = Postprocessor(config.filter, verbose=tracing)
    tracker = Tracker(config.tracker)
    annotator = Annotator()
    tracer = PipelineTracer(args.dropout_dir) if tracing else None
    pipeline = Pipeline(
        camera, display, detector, postprocessor, tracker, annotator, tracer
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
