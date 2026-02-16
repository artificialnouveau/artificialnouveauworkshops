"""
FastAPI backend for ICE Detection (DeepFace ethnicity analysis).

Run:
    cd backend
    pip install -r requirements.txt
    python server.py
"""

import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import uvicorn

app = FastAPI(title="CV Workshop Day 2 — Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded image for faces and ethnicity predictions."""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    try:
        results = DeepFace.analyze(
            img_path=img_array,
            actions=["race"],
            detector_backend="retinaface",
            enforce_detection=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepFace error: {str(e)}")

    # Normalize: DeepFace may return a single dict or a list
    if isinstance(results, dict):
        results = [results]

    faces = []
    for r in results:
        faces.append({
            "region": r.get("region", {}),
            "race": r.get("race", {}),
            "dominant_race": r.get("dominant_race", "unknown"),
        })

    return {"faces": faces, "count": len(faces)}


if __name__ == "__main__":
    print("Starting backend on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
