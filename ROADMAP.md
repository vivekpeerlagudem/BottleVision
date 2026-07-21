# BottleVision — Roadmap

The project is built one milestone at a time. Each milestone produces
something runnable and is reviewed before the next one starts.

Legend: ✅ complete · 🎯 next · 📅 planned · 💭 speculative

---

## ✅ M0 — Project Scaffolding
- Repository, `src/`-layout package, virtual environment, dependencies.
- Empty pipeline "stations" with clear responsibilities.
- Entry point that runs but does nothing.
- **Delivered.** See `docs/architecture.md`.

## ✅ M1 — Webcam Capture + Display Loop
- Camera lifecycle owned by `camera.py` (context-manager based).
- Display window + quit handling in `display.py` (`q` and window close both
  clean).
- Pipeline motor that drives read → show.
- Robust error handling (bad camera index → clear error; dropped frames
  handled).
- **Delivered.** Bug B1 (window close crash) fixed.

## ✅ M2 — Detector on a Single Image
- Pretrained YOLO loaded once, run on a still image via
  `scripts/detect_image.py`.
- `Detection` data contract introduced (framework-agnostic seam).
- Model download managed by `ultralytics`.
- **Delivered.**

## ✅ M3 — Detector in the Live Loop
- Detector wired into the webcam loop; every frame runs inference.
- Annotator draws all detected objects.
- **Delivered.**

## ✅ M4 — Bottle Filtering
- `postprocess.py` filters to a single target class.
- Everything else is dropped.
- **Delivered.**

## ✅ M5 — Multi-Bottle Tracking (+ Horizontal Bottle Support)
- Persistent integer IDs across frames.
- Constant-velocity motion prediction + two-stage association.
- Bootstrap gate for new tracks; full-velocity coasting for lost tracks.
- Class-specific confidence threshold for bottle-equivalent classes.
- `vase` and `wine glass` accepted and relabelled.
- **Delivered — frozen 0.5.0.** See [RELEASE_NOTES_M5.md](RELEASE_NOTES_M5.md)
  and [M5_TEST_REPORT.md](M5_TEST_REPORT.md).

---

## 🎯 M6 — Bottle Recognition & Re-Identification

**Goal:** distinguish *which* bottle. Solves the identity-across-time
limitations of M5's motion-only tracker and enables inventory in M7.

### Objectives
1. **Re-identification (ReID)** — appearance-based identity that survives
   hard swaps, long occlusions, and leave-and-return.
2. **Recognition** — classify a tracked bottle into a persistent product
   identity (e.g. brand / SKU).

### Design (subject to per-milestone approval)

A single primitive — a **crop embedding** — powers both objectives.

New stations:
- **`recognizer.py`** — computes an embedding for each tracked crop.
- **`identity_store.py`** — persists product embeddings + labels; supports
  cold enrolment ("hold up bottle, press key, name it").
- **`Identified`** data contract in `detection.py` — a `Track` plus optional
  product label + similarity.

Modifications:
- **`tracker.py`** — a **rescue-only** appearance path that competes only
  with `spawn` (never overrides high-IoU geometry).
- **`pipeline.py` / `run_webcam.py`** — new stations wired in.
- **`annotator.py`** — shows `#id · brand (sim)` when recognised, else
  `#id · bottle`.
- **`config.py` / `settings.yaml`** — `RecognizerConfig`,
  `IdentityStoreConfig`, similarity thresholds, cold-enrol keybinding.

Not touched: `camera.py`, `display.py`, `geometry.py`, `postprocess.py`,
`detector.py` — the M5 seam holds.

### Sub-milestones
- **M6.1** — embedding station + hard-coded reference; show similarity.
- **M6.2** — ReID rescue in the tracker; measure ID-switch rate reduction.
- **M6.3** — cold-enrolment flow + `IdentityStore` persistence.
- **M6.4** — retrieval evaluation on a small held-out set.

### Explicitly out of scope
- OCR / barcode reading.
- Cloud-hosted model inference.
- Multi-camera fusion.

---

## 📅 M7 — Inventory

**Goal:** turn recognised bottles into a running inventory.

Likely scope:
- SQLite (or equivalent) persistence for product catalogue and stock.
- Enter / exit events per identity (bottle added, bottle removed).
- Simple queries: "how many of each product are on the shelf right now?"
- Session state survives program restarts.

Depends on M6.3 (`IdentityStore`).

---

## 📅 M8 — Point-of-Sale Integration

**Goal:** connect BottleVision to a POS system so a sale automatically
decrements inventory.

Likely scope:
- Minimal REST or IPC endpoint the POS can call ("sold: this SKU").
- Reconciliation logic between vision-observed changes and POS events.
- Audit log.

Depends on M7.

---

## 💭 Post-M8 Ideas (not scheduled)

- **Multi-camera fusion** — combine views from front and shelf-facing cameras.
- **Mobile / edge deployment** — quantised model, GPU or NPU inference.
- **Cloud sync** — push events / recognised identities to a backend.
- **Ordered pick lists / SOP compliance** — verify actions match a
  prescribed sequence.
- **Anomaly detection** — flag unusual movements (theft, breakage).
- **Better model** — retrain on domain-specific bottle imagery to remove the
  reliance on COCO's `bottle` / `vase` split.

---

## Non-Goals (permanent)

Recorded so we do not accidentally scope-creep toward them:

- General-purpose object tracking framework.
- A GUI for training / labelling.
- Replacing a professional POS or inventory system in its full generality.

---

## Version History

| Version | Milestone | Status |
|---:|---|---|
| 0.1.0 | M0 – M4 | Superseded |
| **0.5.0** | **M5** | **Frozen** |
| 0.6.x | M6 sub-milestones | Planned |
| 0.7.0 | M7 | Planned |
| 0.8.0 | M8 | Planned |
