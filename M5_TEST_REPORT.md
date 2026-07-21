# BottleVision — M5 Test Report

**Milestone:** M5 (Multi-Bottle Tracking)
**Report type:** manual acceptance testing on real webcam + end-to-end
integration smoke test on a still image

---

## Testing Approach

M5 acceptance was **manual, human-in-the-loop testing on the real webcam**.
This is the honest form for a stateful real-time system whose failure modes
(motion blur, hand occlusion, lighting, class ambiguity) cannot be faithfully
replayed offline.

Two supporting artefacts backed the manual runs:

- **A frozen still-image end-to-end run** through the complete pipeline
  (detector → filter → tracker → annotator), used as a regression tripwire
  after every code change.
- **Instrumented diagnostic runs** (`m5_evidence*.log`) generated during
  investigation. These are not release artefacts and their instrumentation has
  been removed for the frozen release; the *findings* they produced are cited
  below.

Automated unit tests remain empty placeholders — see limitation #12 in
[docs/known-limitations.md](docs/known-limitations.md).

---

## Environment

| Component | Value |
|---|---|
| OS | Windows 11 |
| Shell | PowerShell (git-bash also available) |
| Python | 3.10+ (project's `.venv`) |
| OpenCV | opencv-python 5.0.0 (HighGUI Win32 backend) |
| Ultralytics YOLO | 8.3.x, `yolov8n.pt` (nano) |
| PyTorch | as pulled by `ultralytics` (CPU) |
| Inference device | **CPU only** |
| Webcam | built-in laptop camera, 640×480 |
| Objects used | white bottle, blue bottle, black bottle, water-cream bottles (upright and horizontal) |

---

## Test Scenarios

Each scenario states the criterion, the expected behaviour, the actual
observed behaviour, and a pass/fail verdict. "PASS" reflects the **final,
frozen** behaviour; where earlier iterations failed, that is noted.

### T1 — Two bottles stationary, no interaction
- **Expected:** two IDs assigned once; both persist for the whole session.
- **Actual:** two IDs assigned, held stable indefinitely across many minutes
  of observation.
- **Result: PASS**

### T2 — Move only the blue bottle, white bottle stationary
- **Expected:** white ID never changes; blue ID never changes.
- **Actual (final):** both IDs preserved across normal hand movement.
- **Actual (earlier iterations):** repeatedly failed with blue `#2 → #7`; the
  root cause was traced to `TRACK EXPIRATION` triggered by 20-frame class-
  filter dropouts where YOLO relabelled the moving bottle as `vase`.
  Fixed by B4 + B5 in [RELEASE_NOTES_M5.md](RELEASE_NOTES_M5.md).
- **Result: PASS**

### T3 — Move only the white bottle
- **Expected:** symmetric with T2.
- **Actual:** as T2; both IDs preserved.
- **Result: PASS**

### T4 — Two bottles moved independently
- **Expected:** each ID follows its own object.
- **Actual:** IDs preserved across independent motion at moderate hand speed.
- **Result: PASS**

### T5 — Two bottles swap positions (smooth arcs, mostly visible)
- **Expected:** if bottles pass each other with visible separation, IDs
  should remain attached to the same physical object.
- **Actual:** smooth crossings with visible separation succeed.
  Hard swaps (physically exchanging positions with mutual occlusion) may
  swap IDs — this is a **documented limit** of the motion-only tracker.
- **Result: PARTIAL PASS** (smooth crossings pass; hard swaps documented as
  known limitation #1, deferred to M6.)

### T6 — Temporary hand occlusion
- **Expected:** bottle reappears with the same ID if occlusion is shorter
  than `max_lost_frames` (≈ 30 frames, ~2–3 seconds at 10 FPS).
- **Actual:** re-acquired with the same ID after brief occlusions in the
  expected window.
- **Result: PASS** (within the documented window)

### T7 — Bottle leaves the frame and returns
- **Expected:** new ID when the bottle returns after leaving.
- **Actual:** new ID assigned on return, as designed. This is not a bug;
  identity across leave/return requires appearance ReID.
- **Result: PASS** (behaviour matches spec; long-term identity deferred to M6)

### T8 — Horizontal bottle detection
- **Expected:** horizontal bottles are tracked with the same ID as when they
  are upright.
- **Actual (final):** horizontal bottles held their IDs. YOLO frequently
  labelled them `vase`; the class-alias fix accepted and relabelled them.
- **Actual (earlier iterations):** horizontal bottles dropped entirely
  because the `bottle` probability fell below the 0.50 threshold. Fixed by
  the class-specific `bottle_confidence_threshold = 0.30`.
- **Result: PASS**

### T9 — Black bottle
- **Expected:** detected and tracked.
- **Actual:** detected reliably; tracking behaves as with any other bottle.
- **Result: PASS**

### T10 — White bottle
- **Expected:** detected and tracked.
- **Actual:** detected reliably; tracking behaves as with any other bottle.
- **Result: PASS**

### T11 — Application quit via **q** key
- **Expected:** camera released, window destroyed, process exits cleanly
  with code 0.
- **Actual:** clean exit, no traceback.
- **Result: PASS**

### T12 — Application quit via window **X** close button
- **Expected:** clean exit, no traceback.
- **Actual (final):** clean exit; log line reads
  `"Display window '...' was already closed by the OS; skipping destroy."`
- **Actual (earlier iterations):** crashed with
  `(-27:Null pointer) NULL window: 'BottleVision'`. Fixed by B1 in the
  release notes.
- **Result: PASS**

### T13 — End-to-end still-image regression (automated smoke)
- **Command:** internal Python one-liner used after every code change
  (see the *End-to-end verification* step of the M5 cleanup).
- **Expected:** on `images/bottle_1.jpeg`, three consecutive calls to the
  full pipeline yield `raw=3 filtered=1 tracks=1` each time.
- **Actual:** exact match, three times in a row.
- **Result: PASS**

### T14 — Configuration validation
- **Expected:** missing / malformed config values raise `ConfigError` with a
  clear message pointing at the specific key.
- **Actual:** verified for each field during development — missing sections,
  wrong types (including `bool` where `int` expected), out-of-range values,
  empty strings.
- **Result: PASS**

---

## Result Summary

| Category | Scenarios | Pass | Partial | Fail |
|---|---:|---:|---:|---:|
| Multi-object tracking | T1–T4 | 4 | 0 | 0 |
| Crossings / swaps | T5 | 0 | 1 | 0 |
| Occlusion / entry / exit | T6, T7 | 2 | 0 | 0 |
| Detection edge cases | T8–T10 | 3 | 0 | 0 |
| Lifecycle / shutdown | T11, T12 | 2 | 0 | 0 |
| Integration / config | T13, T14 | 2 | 0 | 0 |
| **Total** | **14** | **13** | **1** | **0** |

**Zero failing scenarios.** The single PARTIAL is T5, whose failing subset
(hard swap with mutual occlusion) is a **documented, evidence-based
architectural limit** of a motion-only tracker, deferred to M6.

---

## Remaining Edge Cases

These are scenarios the manual test coverage **did not exhaust**. They are
listed for transparency, not as regressions.

1. **Three or more bottles at once.** Tested informally; no principled reason
   to fail (the algorithm scales), but a stress test was not conducted.
2. **Extreme motion** (throwing a bottle across the frame). Not tested; would
   likely exceed distance gates and produce a new ID on landing.
3. **Very small bottles** (far from camera, < 30 px). Not tested; YOLO's
   recall drops on small objects, so intermittent dropouts are expected.
4. **Reflective surfaces / mirrors.** Not tested; a bottle's reflection would
   count as a second detection.
5. **Multiple lighting conditions.** Testing was done under typical indoor
   room lighting only.
6. **Long-running sessions** (> 30 minutes). Not measured. There is no
   obvious mechanism for degradation (the tracker's state is bounded by
   `max_lost_frames`), but memory / stability was not benchmarked.
7. **Programmatic ID overflow.** `_next_id` is an unbounded Python int; not
   a real concern in practice but not tested.
8. **Config edge cases** with unusual valid values (e.g. `max_lost_frames = 0`,
   `iou_threshold = 0.99`). Validation was exercised for bounds; extreme-but-
   valid values were not run end-to-end.

---

## Regression Guard for Future Work

The still-image smoke test (T13) is the cheapest regression tripwire. Any
change to `detector.py`, `postprocess.py`, or `tracker.py` should be
verified to still produce `raw=3 filtered=1 tracks=1` on
`images/bottle_1.jpeg` before it is committed.
