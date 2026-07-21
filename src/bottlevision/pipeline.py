"""Pipeline orchestrator.

Responsibility: wire the stations together and drive the main loop. As of
Milestone M5 the loop is::

    read  ->  detect  ->  filter (bottles)  ->  track (IDs)  ->  annotate  ->  display

The pipeline coordinates the stations but performs no OpenCV or model work
itself. It is GIVEN its stations (dependency injection) rather than creating
them, which keeps it decoupled and easy to test with fakes later.
"""

from __future__ import annotations

from bottlevision.annotator import Annotator
from bottlevision.camera import Camera
from bottlevision.detector import Detector
from bottlevision.diagnostics import PipelineTracer
from bottlevision.display import Display
from bottlevision.postprocess import Postprocessor
from bottlevision.tracker import Tracker
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)

# If the camera drops this many frames in a row, assume it has failed and stop,
# rather than spinning forever on a dead device.
_MAX_CONSECUTIVE_FAILURES = 30


class Pipeline:
    """Runs the capture -> detect -> annotate -> display loop until quit."""

    def __init__(
        self,
        camera: Camera,
        display: Display,
        detector: Detector,
        postprocessor: Postprocessor,
        tracker: Tracker,
        annotator: Annotator,
        tracer: PipelineTracer | None = None,
    ) -> None:
        """Store the already-constructed stations.

        Args:
            camera: The webcam source.
            display: The on-screen window.
            detector: The object detector (its model is already loaded).
            postprocessor: Filters detections down to the target class.
            tracker: Assigns persistent IDs across frames.
            annotator: Draws tracked objects onto frames.
            tracer: Optional diagnostic observer. When supplied, each frame's
                stage-by-stage counts are reported under a single authoritative
                frame number, and frames that reach the tracker with no
                detections are written to disk for visual inspection. Purely
                observational -- no stage behaves differently because of it.
        """
        self._camera = camera
        self._display = display
        self._detector = detector
        self._postprocessor = postprocessor
        self._tracker = tracker
        self._annotator = annotator
        self._tracer = tracer
        # The one authoritative frame number. Stage-local counters could drift
        # apart; this one is incremented exactly once per processed frame.
        self._pipeline_frame = 0

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

                # Detect on every frame (M3), keep only bottles (M4), then give
                # each surviving bottle a persistent identity (M5).
                self._pipeline_frame += 1
                raw_detections = self._detector.detect(frame)
                detections = self._postprocessor.filter(raw_detections)

                # Observe the frame's full journey BEFORE the tracker consumes
                # it, so the trace reflects exactly what the tracker received.
                if self._tracer is not None:
                    self._tracer.trace(
                        frame_number=self._pipeline_frame,
                        frame=frame,
                        detector_trace=self._detector.last_trace,
                        filter_trace=self._postprocessor.last_trace,
                        tracker_received=len(detections),
                    )

                tracks = self._tracker.update(detections)

                annotated = self._annotator.annotate(frame, tracks)

                if not self._display.show(annotated):
                    break

        _log.info("BottleVision stopped.")
