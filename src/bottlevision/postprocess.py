"""Detection post-processing (business logic).

Responsibility: filter the detector's output down to bottle-equivalent classes,
apply the class-specific confidence threshold, and relabel aliases to the
canonical target. This is the ONLY module that knows the app is about bottles;
the detector stays class-agnostic.

Input:  list[Detection]  (any class, above the detector's query threshold)
Output: list[Detection]  (target class only; may be empty)
"""

from __future__ import annotations

from bottlevision.config import FilterConfig
from bottlevision.detection import Detection


class Postprocessor:
    """Keeps only the detections belonging to the configured target class."""

    def __init__(self, config: FilterConfig) -> None:
        """Store the filter policy.

        Args:
            config: Validated filter settings.
        """
        # Compared case-insensitively so a config value like "Bottle" still
        # matches the model's "bottle" label.
        self._target_class = config.target_class.strip().lower()
        # Aliases YOLO commonly picks for the same physical object, kept to
        # avoid dropping a bottle just because it was momentarily called a
        # 'vase' or a 'wine glass'.
        self._accepted_aliases = frozenset(
            name.strip().lower() for name in config.equivalent_classes
        )
        # Class-specific confidence floor for bottle-equivalent detections
        # (typically 0.30) -- lower than the standard threshold to catch
        # horizontal bottles whose probability mass splits between 'bottle' and
        # 'vase'. Non bottle-equivalent classes are dropped by class regardless
        # of confidence.
        self._bottle_confidence_threshold = config.bottle_confidence_threshold

    def filter(self, detections: list[Detection]) -> list[Detection]:
        """Return only the detections matching the target class.

        Args:
            detections: All detections from the detector.

        Returns:
            A new list containing only detections whose class matches the
            target or an accepted alias, and whose confidence meets the
            bottle-equivalent threshold. Aliases are relabelled to the
            canonical target so downstream stages see one consistent class
            name. Empty if none match; the input list is not mutated.
        """
        kept: list[Detection] = []
        for det in detections:
            name = det.class_name.lower()
            # Only bottle-equivalent classes are admitted; everything else is
            # dropped regardless of confidence.
            if name != self._target_class and name not in self._accepted_aliases:
                continue
            if det.confidence < self._bottle_confidence_threshold:
                continue
            if name == self._target_class:
                kept.append(det)
            else:
                # Relabel to the canonical target so the tracker, annotator,
                # and any downstream consumers see one consistent class name.
                kept.append(
                    Detection(
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_id=det.class_id,
                        class_name=self._target_class,
                    )
                )
        return kept
