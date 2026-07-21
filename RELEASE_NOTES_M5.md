# BottleVision — Release Notes: Milestone M5

**Version:** 0.5.0
**Milestone:** M5 — Multi-Bottle Tracking (+ Horizontal Bottle Support)
**Status:** Frozen

---

## Overview

M5 turns BottleVision from a per-frame detector into a **stateful tracker**.
Every bottle in the scene now receives a **persistent integer ID** that stays
attached across normal movement, brief hand occlusion, and short detector
dropouts. As a coupled improvement, YOLO misclassifications of horizontal
bottles as `vase` or `wine glass` are transparently rescued so identity does
not fragment.

The tracker is deliberately **motion-only** (no appearance embedding). Its
limits are honestly documented in [docs/known-limitations.md](docs/known-limitations.md)
and are the natural starting point for M6 (recognition + re-identification).

---

## Features Completed

- **Multi-object tracking** — persistent integer IDs across consecutive frames.
- **Constant-velocity motion model** — each track predicts its next position
  from the exponentially-smoothed velocity of its observations.
- **Two-stage association**
  - Stage 1: highest-IoU pairs (predicted box vs detection), threshold-gated.
  - Stage 2: nearest centre-distance pairs for whatever stage 1 missed, with a
    size-scaled gate.
- **Bootstrap gate widening** — a track with `hits < 2` (no velocity yet) gets
  a 3× wider distance gate so fast-moving objects can reach their second
  observation without spawning a new ID every frame.
- **Full-velocity coasting** — unobserved tracks keep predicting at their last
  known velocity, so a briefly occluded object is re-acquired at the predicted
  position rather than the stale one.
- **Track lifecycle** — CREATE → MATCH → AGE → RETIRE, bounded by
  `max_lost_frames`.
- **Class-specific confidence threshold** — bottle-equivalent classes use
  `bottle_confidence_threshold` (0.30) while everything else still needs the
  standard 0.50.
- **Bottle-equivalent class handling** — `vase` and `wine glass` are accepted
  and **relabelled** to `bottle`, so identity holds when YOLO oscillates.
- **Horizontal bottle detection** — falls out of the two fixes above; no code
  is aware of orientation.

---

## Major Design Decisions

1. **Motion-only, greedy, no Kalman filter.** Simpler, adequate for the
   scenes tested, and empirically **greedy was never the cause** of any real
   ID switch observed (all observed cases were `AMBIGUOUS`, never
   `GREEDY SUBOPTIMAL`). Hungarian assignment is deferred until evidence
   demands it.
2. **Detector stays class-agnostic.** All "bottle" knowledge lives in
   `postprocess.py`. The detector only lowers its YOLO query threshold (via
   `min_query_confidence`) so downstream policy can see borderline detections.
3. **Track relabelling to the canonical target.** A `vase` detection kept via
   `equivalent_classes` reaches the tracker as `class_name="bottle"`. The
   tracker never sees the alias, keeping downstream logic uniform.
4. **Frozen dataclasses for config and data seams** — `Detection`, `Track`,
   `Config` are immutable value types. Autocomplete-friendly, mutation-proof.
5. **Diagnostic instrumentation was removed for release.** Extensive per-frame
   forensic logging (cost matrices, lifecycle timelines, dropout tracing,
   greedy-vs-optimal analysis) was invaluable for isolating M5 bugs but does
   not belong in production. It has been fully excised.

---

## Bugs Fixed During M5

Bugs are listed in the order they were surfaced by real webcam testing, with
the root cause proven from evidence (not guessed).

### B1 — Window close (X button) crashed on exit
**Symptom:** `(-27:Null pointer) NULL window: 'BottleVision'` from
`cv2.destroyWindow`.
**Root cause:** on Win32 HighGUI the native window is destroyed by the
backend *before* our teardown runs; our state flag tracked "we created a
window" not "the window still exists".
**Fix:** guard `destroyWindow` with `getWindowProperty(...) >= 1`
(visibility), the same predicate that already reliably detects the close.

### B2 — Single bottle changed ID every frame while moving (deadlock)
**Symptom:** IDs incrementing `#1 → #2 → #3 → …` on a continuously visible
bottle.
**Root cause:** `_predict()` skipped tracks with `hits < 2` (no velocity
yet), so a fast-moving track could never reach its second observation, so
it could never earn a velocity, so it could never predict — the tight
distance gate kept it forever locked out.
**Fix:** `_BOOTSTRAP_GATE_MULTIPLIER = 3.0` widens the distance gate for
tracks with `hits < 2` only. Established tracks keep the tight gate.

### B3 — Track velocity decayed while unobserved
**Symptom:** IDs still changed after brief detection dropouts even when the
bottle kept moving smoothly.
**Root cause:** velocity was being damped every frame the track was
unobserved, so the predicted box fell progressively behind the real object
and re-acquisition failed once it returned.
**Fix:** removed the damping; lost tracks now coast at full velocity. Verified
safe — a coasting ghost cannot steal a visible bottle (that bottle's own
track scores IoU ≈ 1.0 and wins greedy matching first).

### B4 — 20-frame detection dropouts caused TRACK EXPIRATION → new ID
**Symptom:** blue bottle `#2` becomes `#7` after long dropout.
**Root cause (proven by real log evidence):** 107 / 107 real "no bottle"
frames were classified `LOST AT CLASS FILTER`, not detector dropout. YOLO
saw the object and labelled it `vase` (112 hits), `wine glass`, etc. The
class filter dropped it, the track expired after `max_lost_frames`, and a
new ID was minted on return.
**Fix:** `equivalent_classes: ["vase", "wine glass"]` in `settings.yaml`;
these are accepted and relabelled to `bottle`.

### B5 — Alias fix did not activate; alias detections were below 0.50
**Symptom:** despite B4's fix, ID switches persisted; vase detections
observed at 0.31 / 0.45 / 0.13 were below the confidence gate.
**Root cause:** class-ambiguity dilution — for horizontal / partially
occluded bottles YOLO's probability mass splits between `bottle` and `vase`,
and neither class reaches 0.50 under a single global threshold.
**Fix:** class-specific `bottle_confidence_threshold: 0.30` applied only to
bottle-equivalent classes. Detector queries YOLO at the lower floor so the
postprocessor can see borderline detections; non-equivalent classes are
still dropped regardless of confidence.

### Hypotheses that were investigated and RULED OUT
Documenting these is important — each was suspected but proven wrong by
evidence, and knowing the mechanism is not the cause protects future work.
- **Duplicate detections** (my initial hypothesis) — 4 real switch events
  showed `DUPLICATE: none` in every case. Rejected.
- **Greedy assignment** — over 37 real switches, zero `GREEDY SUBOPTIMAL`
  verdicts; all were `AMBIGUOUS` (ties Hungarian could not resolve either).
  Rejected.
- **Distance gate too strict** — the 215 px displacement in one switch was
  a symptom of a 20-frame dropout, not the root cause. Rejected.
- **Confidence filtering / detector dropout** — real evidence showed YOLO
  was returning detections; the class filter (not confidence) was the
  gate that killed them. Refocused to B4.

---

## Performance Summary

Measurements are qualitative (no formal benchmarking milestone in M5). On
the reference hardware:

| Stage | Per-frame cost (order of magnitude) |
|---|---|
| Camera read | ~1–3 ms |
| YOLO forward pass (`yolov8n`, CPU) | ~40–100 ms |
| Postprocessor (class filter) | < 1 ms |
| Tracker (`update`) | < 1 ms in typical 2-bottle scenes |
| Annotator | ~1 ms |
| Display (`waitKey(1)`) | ~1 ms |

**Effective frame rate is dominated by YOLO** and lands around 10–20 FPS on
CPU. Tracking overhead is negligible. GPU inference would materially change
this but is out of scope for M5.

---

## Known Limitations

Full list with mechanisms and deferred-to milestones in
[docs/known-limitations.md](docs/known-limitations.md). Summary:

- Hard swaps (two bottles physically exchanged, especially mid-occlusion) may
  swap IDs — geometry alone cannot distinguish them. **Deferred to M6 (ReID).**
- Very long occlusions (> `max_lost_frames`, ≈ 30 frames) retire the track and
  assign a new ID on return. Deferred to M6.
- Association is greedy, not globally optimal. Evidence-based deferral —
  revisit only if greedy is proven suboptimal on real data.
- YOLO's default NMS (`iou=0.7`, `agnostic_nms=False`) permits some duplicate
  boxes. Not the dominant failure mode in real testing; detector-side dedup
  was deferred.
- Class-alias set is deliberately small (`vase`, `wine glass`) to avoid the
  false-positive surface of promiscuous classes (`cup`, `person`, `chair`).
- Model cannot identify *which* bottle — COCO's `bottle` is a single generic
  label. **Deferred to M6.**
- Single camera, single session, no persistence between runs. **Deferred to
  M7 (inventory).**
- No point-of-sale integration. **Deferred to M8.**
- Real-time performance is CPU-bound (see Performance Summary).
- Test placeholders remain empty.

---

## Breaking Changes

M5 does not change end-user behaviour of the entry point (`run_webcam.py`
is still parameterless), but the **internal package API** and the
**configuration schema** did change. Consumers building on the library
directly should note:

### Configuration (`config/settings.yaml`)
Two new sections/fields, both with sensible defaults:

```yaml
filter:
  equivalent_classes: ["vase", "wine glass"]   # NEW
  bottle_confidence_threshold: 0.30            # NEW

tracker:                                       # NEW SECTION
  iou_threshold: 0.20
  max_center_distance_factor: 1.20
  velocity_smoothing: 0.50
  max_lost_frames: 30
```

Loading a pre-M5 config file will fail with a clear `ConfigError` at startup
pointing at the missing section.

### Package API
- `bottlevision.detection.Track` — **new** dataclass.
- `bottlevision.tracker.Tracker` — **new** class.
- `bottlevision.postprocess.Postprocessor` — signature unchanged, but the
  `equivalent_classes` and `bottle_confidence_threshold` fields on
  `FilterConfig` are now consulted.
- `bottlevision.detector.Detector.__init__` — gained optional
  `min_query_confidence` parameter (default `None` = previous behaviour).
- `bottlevision.annotator.Annotator.annotate` — signature changed from
  `list[Detection]` to `list[Track]`.
- `bottlevision.pipeline.Pipeline.__init__` — now takes six stations
  (added `postprocessor`, `tracker`).

### Modules removed
- `bottlevision.diagnostics` — the M5 investigation instrumentation module.
- `bottlevision.utils.fps` — never implemented, unused.

---

## Verified On

See [M5_TEST_REPORT.md](M5_TEST_REPORT.md).
