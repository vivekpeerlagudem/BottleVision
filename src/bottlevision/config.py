"""Configuration loading.

Responsibility: read ``config/settings.yaml`` from disk and turn it into a
single, validated, typed configuration object that the rest of the app uses.
If a value is missing or invalid, this module fails loudly and early with a
clear message, rather than letting a bad value cause a cryptic crash later.

The ``camera``, ``display`` (M1), ``detector`` (M2), ``filter`` (M4) and
``tracker`` (M5) sections are loaded here. Because the config is modelled as
dataclasses, adding a section is a small, local change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Resolve the repository's default config file relative to THIS file, so the
# app works no matter which directory it is launched from.
# config.py lives at: <root>/src/bottlevision/config.py  ->  parents[2] == <root>
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


class ConfigError(Exception):
    """Raised when the configuration file is missing, unreadable, or invalid."""


@dataclass(frozen=True)
class CameraConfig:
    """Settings for the webcam source."""

    index: int
    width: int
    height: int


@dataclass(frozen=True)
class DisplayConfig:
    """Settings for the on-screen video window."""

    window_name: str
    show_fps: bool


@dataclass(frozen=True)
class DetectorConfig:
    """Settings for the YOLO object detector."""

    model: str
    confidence_threshold: float


@dataclass(frozen=True)
class FilterConfig:
    """Settings for post-processing (which class of object to keep).

    ``equivalent_classes`` lists COCO labels that should be treated as the
    target class. They exist because pretrained YOLO frequently misclassifies
    bottles -- particularly horizontal ones -- as ``vase`` or ``wine glass``,
    and dropping those detections was measurably responsible for track
    expirations on the real webcam.
    """

    target_class: str
    equivalent_classes: tuple[str, ...] = ()
    bottle_confidence_threshold: float = 0.30


@dataclass(frozen=True)
class TrackerConfig:
    """Settings for multi-object tracking."""

    iou_threshold: float
    max_center_distance_factor: float
    velocity_smoothing: float
    max_lost_frames: int


@dataclass(frozen=True)
class Config:
    """The full, validated application configuration."""

    camera: CameraConfig
    display: DisplayConfig
    detector: DetectorConfig
    filter: FilterConfig
    tracker: TrackerConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load, parse, and validate the YAML configuration file.

    Args:
        path: Path to the YAML file. Defaults to the project's
            ``config/settings.yaml``.

    Returns:
        A fully validated :class:`Config` object.

    Raises:
        ConfigError: If the file is missing, cannot be parsed, or any value
            is missing or of the wrong type.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Configuration root must be a mapping, got {type(raw).__name__}."
        )

    return Config(
        camera=_parse_camera(_section(raw, "camera")),
        display=_parse_display(_section(raw, "display")),
        detector=_parse_detector(_section(raw, "detector")),
        filter=_parse_filter(_section(raw, "filter")),
        tracker=_parse_tracker(_section(raw, "tracker")),
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required top-level section, or raise a clear error."""
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Missing or invalid '{name}' section in configuration.")
    return section


def _require(
    section: dict[str, Any], key: str, expected: type, section_name: str
) -> Any:
    """Fetch ``section[key]``, validating that it exists and has the right type.

    Note: ``bool`` is a subclass of ``int`` in Python, so we explicitly reject a
    boolean where an integer is expected (e.g. ``index: true``).
    """
    if key not in section:
        raise ConfigError(f"Missing '{key}' in '{section_name}' section.")
    value = section[key]
    if not isinstance(value, expected) or (isinstance(value, bool) and expected is int):
        raise ConfigError(
            f"'{section_name}.{key}' must be of type {expected.__name__}, "
            f"got {type(value).__name__}."
        )
    return value


def _parse_camera(section: dict[str, Any]) -> CameraConfig:
    """Validate and build the camera configuration."""
    index = _require(section, "index", int, "camera")
    width = _require(section, "width", int, "camera")
    height = _require(section, "height", int, "camera")

    if index < 0:
        raise ConfigError(f"'camera.index' must be >= 0, got {index}.")
    if width <= 0 or height <= 0:
        raise ConfigError(
            f"'camera.width' and 'camera.height' must be > 0, got {width}x{height}."
        )
    return CameraConfig(index=index, width=width, height=height)


def _parse_display(section: dict[str, Any]) -> DisplayConfig:
    """Validate and build the display configuration."""
    window_name = _require(section, "window_name", str, "display")
    show_fps = _require(section, "show_fps", bool, "display")

    if not window_name.strip():
        raise ConfigError("'display.window_name' must not be empty.")
    return DisplayConfig(window_name=window_name, show_fps=show_fps)


def _parse_detector(section: dict[str, Any]) -> DetectorConfig:
    """Validate and build the detector configuration."""
    model = _require(section, "model", str, "detector")

    # Confidence may be written as ``0.5`` (float) or ``1`` (int); accept both,
    # but reject booleans (``True`` is technically an int in Python).
    if "confidence_threshold" not in section:
        raise ConfigError("Missing 'confidence_threshold' in 'detector' section.")
    conf = section["confidence_threshold"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        raise ConfigError(
            "'detector.confidence_threshold' must be a number, "
            f"got {type(conf).__name__}."
        )
    conf = float(conf)

    if not model.strip():
        raise ConfigError("'detector.model' must not be empty.")
    if not 0.0 <= conf <= 1.0:
        raise ConfigError(
            f"'detector.confidence_threshold' must be between 0 and 1, got {conf}."
        )
    return DetectorConfig(model=model, confidence_threshold=conf)


def _parse_filter(section: dict[str, Any]) -> FilterConfig:
    """Validate and build the filter configuration."""
    target_class = _require(section, "target_class", str, "filter")
    if not target_class.strip():
        raise ConfigError("'filter.target_class' must not be empty.")

    raw_equivalents = section.get("equivalent_classes", [])
    if not isinstance(raw_equivalents, list):
        raise ConfigError(
            "'filter.equivalent_classes' must be a list, got "
            f"{type(raw_equivalents).__name__}."
        )
    equivalents: list[str] = []
    for entry in raw_equivalents:
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigError(
                "'filter.equivalent_classes' must contain only non-empty "
                f"strings, got {entry!r}."
            )
        equivalents.append(entry)
    # Class-specific confidence floor for bottle-equivalent detections.
    # Defaults to 0.30 to catch horizontal bottles that split probability mass
    # between 'bottle' and 'vase'. Everything else still needs the standard
    # detector.confidence_threshold.
    if "bottle_confidence_threshold" in section:
        raw = section["bottle_confidence_threshold"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ConfigError(
                "'filter.bottle_confidence_threshold' must be a number, "
                f"got {type(raw).__name__}."
            )
        bottle_conf = float(raw)
        if not 0.0 <= bottle_conf <= 1.0:
            raise ConfigError(
                "'filter.bottle_confidence_threshold' must be between 0 and 1, "
                f"got {bottle_conf}."
            )
    else:
        bottle_conf = 0.30

    return FilterConfig(
        target_class=target_class,
        equivalent_classes=tuple(equivalents),
        bottle_confidence_threshold=bottle_conf,
    )


def _number(section: dict[str, Any], key: str, section_name: str) -> float:
    """Fetch a required numeric value (int or float), rejecting booleans."""
    if key not in section:
        raise ConfigError(f"Missing '{key}' in '{section_name}' section.")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"'{section_name}.{key}' must be a number, got {type(value).__name__}."
        )
    return float(value)


def _parse_tracker(section: dict[str, Any]) -> TrackerConfig:
    """Validate and build the tracker configuration."""
    iou = _number(section, "iou_threshold", "tracker")
    if not 0.0 <= iou <= 1.0:
        raise ConfigError(
            f"'tracker.iou_threshold' must be between 0 and 1, got {iou}."
        )

    distance_factor = _number(section, "max_center_distance_factor", "tracker")
    if distance_factor < 0.0:
        raise ConfigError(
            "'tracker.max_center_distance_factor' must be >= 0, "
            f"got {distance_factor}."
        )

    smoothing = _number(section, "velocity_smoothing", "tracker")
    if not 0.0 <= smoothing <= 1.0:
        raise ConfigError(
            f"'tracker.velocity_smoothing' must be between 0 and 1, got {smoothing}."
        )

    max_lost_frames = _require(section, "max_lost_frames", int, "tracker")
    if max_lost_frames < 0:
        raise ConfigError(
            f"'tracker.max_lost_frames' must be >= 0, got {max_lost_frames}."
        )

    return TrackerConfig(
        iou_threshold=iou,
        max_center_distance_factor=distance_factor,
        velocity_smoothing=smoothing,
        max_lost_frames=max_lost_frames,
    )
