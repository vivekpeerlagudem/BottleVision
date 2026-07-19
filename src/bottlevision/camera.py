"""Webcam source.

Responsibility: own the webcam lifecycle (open, read frames, release) and be
the ONLY module that knows how the camera hardware is accessed. It hands out
plain image frames; nothing else needs to know OpenCV's capture API exists.

``Camera`` is a context manager, so the hardware is always released -- even if
an error occurs mid-loop::

    with Camera(config) as camera:
        frame = camera.read()
"""

from __future__ import annotations

from types import TracebackType

import cv2
import numpy as np

from bottlevision.config import CameraConfig
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or used."""


class Camera:
    """A thin, safe wrapper around an OpenCV video capture device."""

    def __init__(self, config: CameraConfig) -> None:
        """Store configuration. The device is NOT opened until :meth:`open`.

        Args:
            config: Validated camera settings (index and desired resolution).
        """
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the webcam and apply the requested resolution.

        Raises:
            CameraError: If the device at the configured index cannot be opened.
        """
        capture = cv2.VideoCapture(self._config.index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera at index {self._config.index}. "
                "Is it connected and not in use by another application?"
            )

        # These are requests: the driver picks the nearest supported size.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._capture = capture

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _log.info(
            "Camera opened (index=%d, requested %dx%d, actual %dx%d).",
            self._config.index,
            self._config.width,
            self._config.height,
            actual_w,
            actual_h,
        )

    def read(self) -> np.ndarray | None:
        """Read the next frame from the camera.

        Returns:
            The frame as a BGR image array of shape ``(height, width, 3)``, or
            ``None`` if this particular grab failed (a transient dropped frame).

        Raises:
            CameraError: If called before :meth:`open`.
        """
        if self._capture is None:
            raise CameraError("Camera.read() called before open().")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            _log.warning("Dropped a frame from the camera.")
            return None
        return frame

    def release(self) -> None:
        """Release the camera hardware. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            _log.info("Camera released.")

    def __enter__(self) -> Camera:
        """Open the camera on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always release the camera when leaving a ``with`` block."""
        self.release()
