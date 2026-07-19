"""Display / window.

Responsibility: show an annotated frame in a window and report whether the user
asked to quit -- either by pressing 'q' or by closing the window. The only
module that owns the on-screen window and keyboard handling.

``Display`` is a context manager, so the window is always destroyed on exit::

    with Display(config) as display:
        keep_going = display.show(frame)
"""

from __future__ import annotations

from types import TracebackType

import cv2
import numpy as np

from bottlevision.config import DisplayConfig
from bottlevision.utils.logging import get_logger

_log = get_logger(__name__)

_QUIT_KEY = ord("q")


class Display:
    """A thin, safe wrapper around a single OpenCV display window."""

    def __init__(self, config: DisplayConfig) -> None:
        """Store configuration. The window is created in :meth:`open`.

        Args:
            config: Validated display settings (window name, etc.).
        """
        self._config = config
        self._window = config.window_name
        self._created = False

    def open(self) -> None:
        """Create the display window."""
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        self._created = True
        _log.info("Display window '%s' created.", self._window)

    def show(self, frame: np.ndarray) -> bool:
        """Display one frame and check whether the user wants to keep going.

        Args:
            frame: A BGR image array to display.

        Returns:
            ``True`` to continue the loop, ``False`` if the user pressed 'q'
            or closed the window.
        """
        cv2.imshow(self._window, frame)

        # waitKey both renders the window and reads the keyboard. The 1 ms wait
        # is what makes the video appear "live"; it must be called every frame.
        key = cv2.waitKey(1) & 0xFF
        if key == _QUIT_KEY:
            _log.info("Quit requested via 'q' key.")
            return False

        if self._is_window_closed():
            _log.info("Quit requested via window close button.")
            return False

        return True

    def _is_window_closed(self) -> bool:
        """Return True if the window is no longer open (user clicked 'X').

        Uses ``WND_PROP_VISIBLE``: a live, shown window reports 1; once the
        native window is gone it reports 0 (or -1). This single predicate serves
        two purposes -- as the quit signal in :meth:`show`, and as the guard in
        :meth:`close` that prevents destroying an already-dead window.
        """
        try:
            # WND_PROP_VISIBLE drops below 1 once the window has been closed.
            return cv2.getWindowProperty(self._window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            # The window no longer exists; treat as closed.
            return True

    def close(self) -> None:
        """Destroy the display window, but only while it is still open. Idempotent.

        Root cause this guards against (Win32 HighGUI): clicking the window's
        'X' makes the backend destroy the *native* window (HWND) asynchronously,
        via the message loop that ``cv2.waitKey`` pumps. For a brief moment
        afterwards OpenCV's internal registry node for the window still exists
        while its HWND is already NULL. ``cv2.destroyWindow`` finds that
        lingering node and calls the backend with a NULL handle, raising
        "NULL window" in ``destroyWindowImpl``.

        The reliable signal is *visibility*: ``WND_PROP_VISIBLE`` is 1 only for a
        live, shown window and 0 (or -1) once the native window is gone. We
        destroy only when the window is still open -- reusing the very same
        ``_is_window_closed`` predicate ``show()`` uses to detect the close.
        Because nothing pumps the event loop between ``show()`` detecting the
        close and this call, the two observations are guaranteed consistent, so
        we never hand ``destroyWindow`` a dead handle. ``destroyWindow`` itself
        is left unguarded, so any unrelated failure still surfaces.
        """
        if not self._created:
            return
        self._created = False

        if not self._is_window_closed():
            cv2.destroyWindow(self._window)
            _log.info("Display window destroyed.")
        else:
            _log.info(
                "Display window '%s' was already closed by the OS; skipping destroy.",
                self._window,
            )

    def __enter__(self) -> Display:
        """Create the window on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always destroy the window when leaving a ``with`` block."""
        self.close()
