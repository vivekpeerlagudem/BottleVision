"""Detection post-processing (business logic).

Responsibility: filter the detector's raw output down to what we care about --
keep only detections whose class matches the configured target (``"bottle"``).
This is the ONLY module that knows the app is about bottles; the detector stays
class-agnostic.

Input:  list[Detection]  (all classes)
Output: list[Detection]  (target class only; may be empty)
"""

from __future__ import annotations

from bottlevision.config import FilterConfig
from bottlevision.detection import Detection


class Postprocessor:
    """Keeps only the detections belonging to the configured target class."""

    def __init__(self, config: FilterConfig) -> None:
        """Store the target class.

        Args:
            config: Validated filter settings (which class to keep).
        """
        # Compared case-insensitively so a config value like "Bottle" still
        # matches the model's "bottle" label.
        self._target_class = config.target_class.strip().lower()

    def filter(self, detections: list[Detection]) -> list[Detection]:
        """Return only the detections matching the target class.

        Args:
            detections: All detections from the detector.

        Returns:
            A new list containing only detections whose ``class_name`` matches
            the target class. Empty if none match. The input list is not
            mutated, and the relative order of matches is preserved.
        """
        return [
            det for det in detections if det.class_name.lower() == self._target_class
        ]
