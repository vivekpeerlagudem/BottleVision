# BottleVision

Real-time bottle detection from a laptop webcam, built as a learning project.

> **Prototype v0.1** — detect bottles in a live webcam feed using a pre-trained
> YOLO model, draw bounding boxes, and show confidence scores. Nothing more (yet).

---

## What this is

BottleVision is a small, deliberately simple computer-vision application. A frame
travels through a one-way **pipeline** of independent stations:

```
webcam → detector → filter (bottles only) → annotator → display → (loop)
```

Each station does exactly one job. This makes the code easy to read, easy to test,
and easy to extend later (tracking, a database, etc.) without rewriting everything.

## Out of scope for v0.1

Recognition of specific bottles, OCR, barcodes, tracking, databases, inventory,
POS integration, cloud, and mobile. These may come in later versions.

---

## Requirements

- Python 3.10 or newer
- A working webcam
- The dependencies listed in `requirements.txt`

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the project itself (editable mode)
pip install -e .
```

## Run

```bash
python scripts/run_webcam.py
```

> The webcam pipeline is not implemented yet (we are at Milestone M0:
> project scaffolding). Running the entry point currently does nothing.

## Configuration

All tunable values live in `config/settings.yaml`. You can change the camera
index, confidence threshold, and target class **without editing any Python code**.

---

## Project layout

```
BottleVision/
├── config/settings.yaml       # All tunable values
├── src/bottlevision/          # The application package (the pipeline stations)
├── scripts/run_webcam.py      # Entry point you run
├── tests/                     # Automated tests
└── docs/architecture.md       # Design document
```

See `docs/architecture.md` for the full architecture and the milestone plan.

## Development

```bash
pytest        # run the test suite
```
