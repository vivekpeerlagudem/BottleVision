# Known Limitations (as of M5)

Behaviours that are **intentionally deferred**. Each entry names the trigger,
the mechanism, and the milestone (or design change) that would address it.

---

## Tracking

### 1. Hard swaps and heavy mutual occlusion may exchange IDs

**Trigger:** two bottles are lifted and placed in each other's positions, or
they occlude each other for long enough that the motion prediction becomes
meaningless.
**Mechanism:** the tracker is motion-only. When two objects are geometrically
indistinguishable at the crossing point, no motion model can preserve identity.
**Deferred to:** M6 (appearance-based re-identification).

### 2. Very long occlusions retire the track

**Trigger:** a bottle is out of view or undetected for more than
`max_lost_frames` frames (default 30, ≈ 2–3 seconds).
**Mechanism:** by design — after that the coasted position is untrustworthy
and the ID is retired. If the same bottle reappears it receives a new ID.
**Deferred to:** M6 (ReID can rejoin the identity from appearance alone).

### 3. Association is greedy, not globally optimal

**Trigger:** dense multi-bottle scenes where two tracks and two detections all
score similarly.
**Mechanism:** stage-1 IoU and stage-2 centre distance both pick highest-score
pairs first. On the real webcam, greedy was measurably not the cause of any
observed ID switch (verdict: `AMBIGUOUS`, not `GREEDY SUBOPTIMAL`), so the
Hungarian upgrade was deferred.
**Deferred to:** revisit only if evidence shows greedy is the culprit.

## Detector

### 4. YOLO's default NMS lets some duplicates survive

**Trigger:** two overlapping boxes for the same object at IoU < 0.7, or
different classes at the same location (`agnostic_nms=False`).
**Mechanism:** Ultralytics defaults are `iou=0.7` (permissive) and
`agnostic_nms=False` (per-class competitions). Duplicates were not the
dominant cause of ID switches in real testing, so detector-level dedup was
not implemented.
**Deferred to:** implement only if duplicates become a measurable problem
after lowering the bottle threshold.

### 5. Class-alias set is minimal on purpose

**Trigger:** a bottle labelled by YOLO as something other than
`bottle` / `vase` / `wine glass`.
**Mechanism:** promiscuous classes (`cup`, `person`, `chair`, `cell phone`)
were explicitly excluded from `equivalent_classes` because they match many
non-bottle objects and would flood the tracker with false positives.
**Deferred to:** revisit when a specific misclassification is observed in
practice; add the class after confirming its false-positive rate.

### 6. Confidence and NMS tuning is fixed per class

**Trigger:** deployments where the confidence tradeoff or NMS aggressiveness
should differ per class (e.g. a stricter bottle threshold in cluttered scenes).
**Mechanism:** the current pipeline exposes one `confidence_threshold` and
one `bottle_confidence_threshold`. NMS defaults to Ultralytics'.
**Deferred to:** M6+ when the additional signal (appearance) motivates per-
class control.

## Model

### 7. Pretrained COCO YOLO cannot identify *which* bottle

**Trigger:** any need to tell one bottle from another (brand, contents, size).
**Mechanism:** COCO's `bottle` class is a single generic label. The model
outputs "there is a bottle here", not "this is a Coke bottle".
**Deferred to:** M6 (recognition — likely a separate classifier or embedding
model on the tracked crop).

### 8. No OCR, no barcode, no label reading

**Trigger:** any need to extract text or product codes from the bottle.
**Mechanism:** deliberately out of scope for the current pipeline.
**Deferred to:** M6+ (recognition tier, if the strategy requires text).

## Application

### 9. Single camera, single session, no persistence

**Trigger:** anything that needs history between runs.
**Mechanism:** IDs are integers that reset each time the program starts. The
tracker keeps no state on disk.
**Deferred to:** M7 (inventory).

### 10. No point-of-sale integration

**Trigger:** any downstream commercial action.
**Deferred to:** M8.

### 11. Real-time performance is CPU-bound

**Trigger:** any machine without a GPU.
**Mechanism:** YOLO's forward pass dominates the per-frame budget on CPU
(~40–100 ms/frame with `yolov8n`), capping the effective frame rate at
roughly 10–20 FPS. Tracking overhead is negligible by comparison.
**Deferred to:** performance milestone — GPU inference or a smaller /
quantised model. Not required for prototype functionality.

## Tests

### 12. Test placeholders remain empty

**Trigger:** `pytest` runs and reports "no tests collected."
**Mechanism:** the `tests/` files are documented stubs from M0. The most
valuable one to fill first is `test_postprocess.py` (pure logic, no hardware).
**Deferred to:** a dedicated testing milestone once the API is stable
post-M6.
