# BottleVision — Architecture (as of M5)

## Core idea

A real-time CV app is a **one-way pipeline**. A frame enters at one end and a
rendered image leaves the other. Each module is one station on the belt and does
exactly one thing.

```
webcam → detector → filter (bottles) → tracker (IDs) → annotator → display → (loop)
```

## Module responsibilities

| Module | Single responsibility |
|---|---|
| `config.py` | Load & validate `settings.yaml` into one typed object |
| `camera.py` | Own the webcam; hand out frames |
| `detector.py` | Run YOLO on a frame; return raw detections (class-agnostic) |
| `postprocess.py` | Keep only bottle-equivalent classes; relabel to `bottle` |
| `tracker.py` | Assign and preserve persistent integer IDs across frames |
| `annotator.py` | Draw boxes + `#id class conf` labels onto a frame |
| `display.py` | Show the window; detect the quit key / window close |
| `pipeline.py` | Wire the stations together and drive the loop |
| `geometry.py` | Small pure-function box math (IoU, centre, mean side) |
| `detection.py` | Data contracts: `Detection` and `Track` |
| `utils/logging.py` | One consistent logging setup |

## Key design decisions

1. **The detector does not know what a bottle is.** Only `postprocess.py` does.
   To detect other objects later, change one module.
2. **Detections travel as a small explicit data structure**
   (`bbox`, `confidence`, `class_name`), not raw model tensors. This clean seam
   keeps modules swappable and testable.
3. **Tracks are a separate contract from detections** (`Track` adds
   `track_id`), so the annotator depends on the *type* and not on any specific
   tracker.
4. **Config is data, not code.** Tune behaviour via YAML.
5. **Design for testability.** Pure logic (filtering, config, IoU, drawing) is
   isolated from hardware (camera) and the model.

## Data shapes

- **Frame** — NumPy array `(height, width, 3)`, in **BGR** channel order (OpenCV).
- **Detection** — box `(x1, y1, x2, y2)` in pixels + class name + confidence `0–1`.
- **Track** — a `Detection` plus a stable integer `track_id`.
- **Annotated frame** — same array shape, with boxes/labels drawn on.

## The tracker (M5 highlight)

A constant-velocity motion model with two-stage greedy association:

```
per frame:
  predict:   each track's box shifts by its estimated velocity
  associate:
    stage 1: highest IoU pairs (predicted box vs detection), threshold-gated
    stage 2: nearest centre-distance pairs for the rest, size-scaled gate
  update:    matched tracks adopt the observation, refine velocity (EMA)
  age:       unmatched tracks +1 to frames_since_seen
  spawn:     unmatched detections → new track ID
  retire:    tracks unseen for more than max_lost_frames
  report:    return only tracks confirmed in this frame
```

Cross-boundary decisions worth naming:

- **Bootstrap gate widening.** A track with `hits < 2` has no velocity yet, so
  its distance gate is widened (×3) — without this, fast-moving objects can
  never reach a second observation, so they never earn a velocity, so they
  never predict, so every frame spawns a fresh ID. This was the deadlock
  behind the *"single bottle keeps getting a new ID"* regression.
- **Lost tracks keep coasting at full velocity.** An earlier fix damped the
  velocity while unobserved and caused the predicted box to fall progressively
  behind the real object. Coasting is bounded by `max_lost_frames`, and a
  coasting track cannot steal a visible bottle because that bottle's own track
  scores far higher IoU.
- **Class-specific confidence threshold** in `postprocess.py`. YOLO's
  probability mass splits between `bottle` and `vase` for horizontal bottles,
  so neither class reaches the standard 0.50. A separate
  `bottle_confidence_threshold` (0.30) applies to bottle-equivalent classes
  only, while everything else still needs 0.50. The detector queries YOLO at
  the lower floor so the postprocessor can see borderline detections.
- **Bottle-equivalent class list.** COCO labels `vase` and `wine glass` are
  accepted and **relabelled** to `bottle`, keeping identity through frames
  where YOLO happens to prefer the alias.

## Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Project scaffolding | ✅ |
| M1 | Webcam capture + display loop | ✅ |
| M2 | Detector on a single image | ✅ |
| M3 | Detector in the live loop | ✅ |
| M4 | Bottles-only filter | ✅ |
| M5 | Multi-bottle tracking (+horizontal bottle support) | ✅ |
| M6 | Bottle recognition / re-identification | Planned |
| M7 | Inventory | Planned |
| M8 | POS integration | Planned |

## M5 completion summary

Delivered:
- Stable multi-bottle tracking with persistent integer IDs.
- IDs survive normal movement, brief hand occlusion, and detector dropouts of
  up to `max_lost_frames` frames.
- Horizontal bottle detection via bottle-equivalent classes + class-specific
  confidence threshold.
- White and black bottle detection verified on the real webcam.

The tracker is deliberately motion-only (no appearance embedding). Its known
limits are documented in [known-limitations.md](known-limitations.md) and are
the natural starting point for M6.
