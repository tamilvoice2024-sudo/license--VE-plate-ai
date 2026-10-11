# License Plate Recognition AI — Cloud Starter

This starter project contains an English-language browser UI and a Python API backend.

## Important limitations
- The project does not include trained model weights.
- A trained license-plate detector must be placed at `models/plate_detector.pt`.
- Vehicle tracking and multi-frame OCR consensus are not fully implemented in this starter.
- Do not treat OCR confidence as a guarantee that a plate number is correct.

## Project layout
- `backend/main.py` — FastAPI backend for video upload, plate detection, OCR, and CSV export
- `web/index.html` — English-language website
- `dataset/plate_detection/` — YOLO dataset folders
- `models/` — model weights folder

## Dataset format
Place images in:
```
dataset/plate_detection/images/train/
dataset/plate_detection/images/val/
```
Place YOLO bounding-box labels in:
```
dataset/plate_detection/labels/train/
dataset/plate_detection/labels/val/
```
Each label file has the same basename as its image and contains:
`class_id x_center y_center width height`
Coordinates must be normalized to 0–1. For a single-class plate detector, use class ID `0`.

## Install
Use a Linux cloud server or GPU notebook for model training. On the server:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Train a detector
After labels and `data.yaml` are ready:
```
yolo detect train model=yolo11n.pt data=dataset/plate_detection/data.yaml epochs=80 imgsz=640
```
Copy the resulting best weights to `models/plate_detector.pt`, then restart the backend.
Model/version availability depends on your installed Ultralytics version. Validate before production use.

## Environment variables
- `PLATE_MODEL_PATH` (default `models/plate_detector.pt`)
- `VEHICLE_MODEL_PATH` (optional; default `models/vehicle_detector.pt`)
- `MAX_VIDEO_MB` (default `500`)

## API endpoints
- `GET /api/health`
- `POST /api/analyze` — upload a video and download a CSV report
- `GET /` — English web UI

## Current behavior
The API samples video frames at up to 3 frames/second, detects plates using the configured plate model, applies OCR to each detected crop, and returns CSV rows. It does not yet assign persistent vehicle tracking IDs. Add a tracker and multi-frame plate consensus as a subsequent step.
