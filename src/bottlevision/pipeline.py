"""Pipeline orchestrator.

Responsibility: wire the stations together and drive the main loop. The loop is:

    read a frame -> detect -> filter (bottles) -> track (IDs) -> annotate -> display

The pipeline coordinates the stations but performs no OpenCV or model work
itself. Stations are injected rather than constructed here, which keeps the
pipeline decoupled and easy to test with fakes.
"""

from __future__ import annotations

from bottlevision.annotator import Annotator
from bottlevision.camera import Camera
from bottlevision.detector import Detector
from bottlevision.display import Display
from bottlevision.postprocess import Postprocessor
from bottlevision.tracker import Tracker
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)

# If the camera drops this many frames in a row, assume it has failed and stop
# rather than spinning forever on a dead device.
_MAX_CONSECUTIVE_FAILURES = 30


class Pipeline:
    """Runs the capture -> detect -> filter -> track -> annotate -> display loop."""

    def __init__(
        self,
        camera: Camera,
        display: Display,
        detector: Detector,
        postprocessor: Postprocessor,
        tracker: Tracker,
        annotator: Annotator,
    ) -> None:
        """Store the already-constructed stations."""
        self._camera = camera
        self._display = display
        self._detector = detector
        self._postprocessor = postprocessor
        self._tracker = tracker
        self._annotator = annotator

    def run(self) -> None:
        """Open the stations and run the loop until the user quits.

        The camera and display are opened via their context managers, so they
        are guaranteed to be released/destroyed on exit -- even if an error is
        raised inside the loop.
        """
        _log.info("Starting BottleVision. Press 'q' or close the window to quit.")
        consecutive_failures = 0

        with self._camera, self._display:
            while True:
                frame = self._camera.read()

                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        _log.error(
                            "Camera produced no frame %d times in a row; stopping.",
                            consecutive_failures,
                        )
                        break
                    continue
                consecutive_failures = 0

                # Detect on every frame, keep only bottles, then give each
                # surviving bottle a persistent identity.
                detections = self._detector.detect(frame)
                detections = self._postprocessor.filter(detections)
                tracks = self._tracker.update(detections)

                annotated = self._annotator.annotate(frame, tracks)

                if not self._display.show(annotated):
                    break

        _log.info("BottleVision stopped.")
