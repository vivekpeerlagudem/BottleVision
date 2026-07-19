# BottleVision — Architecture (Prototype v0.1)

## Core idea

A real-time CV app is a **one-way pipeline**. A frame enters at one end and a
rendered image leaves the other. Each module is one station on the belt and does
exactly one thing.

```
webcam → detector → filter (bottles only) → annotator → display → (loop)
```

## Module responsibilities

| Module | Single responsibility |
|---|---|
| `config.py` | Load & validate `settings.yaml` into one typed object |
| `camera.py` | Own the webcam; hand out frames |
| `detector.py` | Run YOLO on a frame; return raw detections (all classes) |
| `postprocess.py` | Keep only bottles above the confidence threshold |
| `annotator.py` | Draw boxes + confidence labels onto a frame |
| `display.py` | Show the window; detect the quit key |
| `pipeline.py` | Wire stations together and drive the loop |
| `utils/logging.py` | Consistent logging |
| `utils/fps.py` | Measure loop speed |

## Key design decisions

1. **The detector does not know what a bottle is.** Only `postprocess.py` does.
   To detect other objects later, change one module.
2. **Detections travel as a small explicit data structure**
   (`bbox`, `confidence`, `class_name`), not raw model tensors. This clean seam
   keeps modules swappable and testable.
3. **Config is data, not code.** Tune behavior via YAML.
4. **Design for testability.** Pure logic (filtering, config, drawing) is
   isolated from hardware (camera) and the model, so it can be unit-tested.

## Data shapes

- **Frame** — NumPy array `(height, width, 3)`, in **BGR** channel order (OpenCV).
- **Detection** — box `(x1, y1, x2, y2)` in pixels + class name + confidence `0–1`.
- **Annotated frame** — same array shape, with boxes/labels drawn on.

## Milestones

| # | Milestone |
|---|---|
| M0 | Project scaffolding *(current)* |
| M1 | Webcam capture + display loop |
| M2 | Detector on a single image |
| M3 | Detector in the live loop |
| M4 | Bottles-only filter |
| M5 | Draw boxes + confidence |
| M6 | Config + FPS + polish |
| M7 | Tests |

## Testing strategy

- **Unit tests** for pure logic: `postprocess`, `config`, `annotator`.
- **Integration test** for the detector using committed sample images.
- **Manual smoke test** for the webcam loop (a documented checklist).
