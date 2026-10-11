import os, tempfile
from pathlib import Path
import cv2
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
MODELS = ROOT / "models"
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", str(MODELS / "plate_detector.pt"))
VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", str(MODELS / "vehicle_detector.pt"))
MAX_VIDEO_MB = int(os.getenv("MAX_VIDEO_MB", "500"))

app = FastAPI(title="License Plate Recognition API")
app.mount("/static", StaticFiles(directory=WEB), name="static")
plate_model = None
vehicle_model = None
ocr_engine = None

@app.on_event("startup")
def load_models():
    global plate_model, vehicle_model, ocr_engine
    if Path(PLATE_MODEL_PATH).exists():
        plate_model = YOLO(PLATE_MODEL_PATH)
    if Path(VEHICLE_MODEL_PATH).exists():
        vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    try:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception as e:
        print("OCR initialization failed:", e)

@app.get("/", response_class=HTMLResponse)
def home():
    return (WEB / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "plate_model_loaded": plate_model is not None,
        "vehicle_model_loaded": vehicle_model is not None,
        "ocr_loaded": ocr_engine is not None,
        "message": "Add a trained plate detector at models/plate_detector.pt before analyzing videos."
    }

def run_ocr(crop):
    if ocr_engine is None or crop is None or crop.size == 0:
        return "", 0.0
    h, w = crop.shape[:2]
    crop = cv2.resize(crop, (max(1, w*3), max(1, h*3)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    try:
        result = ocr_engine.ocr(gray, cls=True)
        best_text, best_conf = "", 0.0
        for page in (result or []):
            for line in (page or []):
                if len(line) >= 2 and line[1]:
                    txt, conf = line[1][0], float(line[1][1])
                    txt = "".join(ch for ch in txt.upper() if ch.isalnum())
                    if txt and conf > best_conf:
                        best_text, best_conf = txt, conf
        return best_text, best_conf
    except Exception:
        return "", 0.0

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(400, "Upload a supported video file (MP4, AVI, MOV, MKV, WEBM).")
    if plate_model is None:
        raise HTTPException(503, "Plate detector model is not loaded. Add models/plate_detector.pt and restart the API.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".mp4")
    try:
        total = 0
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_VIDEO_MB * 1024 * 1024:
                raise HTTPException(413, f"Video exceeds the {MAX_VIDEO_MB} MB limit.")
            tmp.write(chunk)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise HTTPException(400, "Could not open the uploaded video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, round(fps / 3))
        rows, frame_idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                result = plate_model.predict(frame, verbose=False, conf=0.20, imgsz=960)[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        text, ocr_conf = run_ocr(frame[y1:y2, x1:x2])
                        rows.append({
                            "Timestamp (seconds)": round(frame_idx / fps, 3),
                            "Frame Number": frame_idx,
                            "Plate Number (OCR)": text,
                            "OCR Confidence": round(ocr_conf, 4),
                            "Plate Detection Confidence": round(float(box.conf[0]), 4),
                            "Plate Crop Coordinates": f"{x1},{y1},{x2},{y2}",
                            "Read Status": "Read" if text else "Plate detected; unreadable"
                        })
            frame_idx += 1
        cap.release()
        if not rows:
            rows.append({
                "Timestamp (seconds)": "", "Frame Number": "",
                "Plate Number (OCR)": "", "OCR Confidence": "",
                "Plate Detection Confidence": "", "Plate Crop Coordinates": "",
                "Read Status": "No plate detections. Check the model, video quality, and confidence threshold."
            })
        out = tempfile.SpooledTemporaryFile(mode="w+", newline="", encoding="utf-8")
        pd.DataFrame(rows).to_csv(out, index=False)
        out.seek(0)
        out_name = Path(file.filename).stem + "_plate_report.csv"
        return StreamingResponse(out, media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'})
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
