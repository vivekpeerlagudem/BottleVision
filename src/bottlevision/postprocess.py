"""Detection post-processing (business logic).

Responsibility: filter the detector's raw output down to what we care about --
keep only detections whose class matches the configured target (``"bottle"``).
This is the ONLY module that knows the app is about bottles; the detector stays
class-agnostic.

Input:  list[Detection]  (all classes)
Output: list[Detection]  (target class only; may be empty)
"""

from __future__ import annotations

from collections import Counter

from bottlevision.config import FilterConfig
from bottlevision.detection import Detection
from bottlevision.diagnostics import FilterTrace


class Postprocessor:
    """Keeps only the detections belonging to the configured target class."""

    def __init__(self, config: FilterConfig, verbose: bool = False) -> None:
        """Store the target class.

        Args:
            config: Validated filter settings (which class to keep).
            verbose: When true, record every frame's filtering decision on
                ``last_trace`` for the pipeline to report. This makes it visible
                when the detector saw the bottle but labelled it something else
                (``cup``, ``vase``, ``wine glass`` ...), which is indistinguishable
                from a detector dropout when viewed from the tracker's side.
        """
        # Compared case-insensitively so a config value like "Bottle" still
        # matches the model's "bottle" label.
        self._target_class = config.target_class.strip().lower()
        # Aliases that YOLO commonly picks for the same physical object, kept
        # to avoid dropping a bottle just because it was momentarily called a
        # 'vase' or a 'wine glass'.
        self._accepted_aliases = frozenset(
            name.strip().lower() for name in config.equivalent_classes
        )
        self._accepted = self._accepted_aliases | {self._target_class}
        # Class-specific confidence floor: bottle-equivalent detections need to
        # pass THIS threshold (0.30 by default) rather than the detector's
        # standard 0.50. It exists because YOLO's probability mass splits
        # between 'bottle' and 'vase' for horizontal bottles, so neither class
        # reaches 0.50 even though the object is confidently detected.
        self._bottle_confidence_threshold = config.bottle_confidence_threshold
        self._verbose = verbose
        # Populated per frame in diagnostic mode; None in normal operation.
        self.last_trace: FilterTrace | None = None

    def filter(self, detections: list[Detection]) -> list[Detection]:
        """Return only the detections matching the target class.

        Args:
            detections: All detections from the detector.

        Returns:
            A new list containing only detections whose ``class_name`` matches
            the target class. Empty if none match. The input list is not
            mutated, and the relative order of matches is preserved.
        """
        kept: list[Detection] = []
        for det in detections:
            name = det.class_name.lower()
            # Non-bottle-equivalent classes are dropped entirely regardless of
            # confidence: the standard 0.50 threshold has effectively already
            # been enforced by the detector, and we do not admit those classes
            # here even if they were to pass it.
            if name != self._target_class and name not in self._accepted_aliases:
                continue
            # Bottle-equivalent classes use the lower, class-specific floor.
            if det.confidence < self._bottle_confidence_threshold:
                continue
            if name == self._target_class:
                kept.append(det)
            else:
                # Relabel to the canonical target so the tracker, annotator,
                # and any downstream consumers see one consistent class name
                # even when YOLO oscillates between 'bottle' and 'vase'.
                kept.append(
                    Detection(
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_id=det.class_id,
                        class_name=self._target_class,
                    )
                )

        if self._verbose:
            self.last_trace = FilterTrace(
                target_class=self._target_class,
                received=len(detections),
                kept=len(kept),
                dropped=Counter(
                    det.class_name
                    for det in detections
                    if det.class_name.lower() != self._target_class
                ),
            )

        return kept
