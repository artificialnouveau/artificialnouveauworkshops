"""
Google Vision & Gemini API proxy server for the CV Workshop.

Keeps API keys server-side so students never see them.
Enforces separate hard request caps for each API to stay within budget.

Setup:
    cd backend
    pip install -r requirements.txt
    export GOOGLE_VISION_API_KEY="your-key-here"
    export GEMINI_API_KEY="your-key-here"
    python server.py

Optional env vars:
    MAX_VISION_REQUESTS  — Vision API lifetime cap (default: 6000, ~$9 at $1.50/1K)
    MAX_GEMINI_REQUESTS  — Gemini API lifetime cap (default: 1500, ~$9 at Flash pricing)
    PORT                 — server port (default: 8000)
"""

import base64
import io
import json
import os
import threading

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MAX_VISION_REQUESTS = int(os.environ.get("MAX_VISION_REQUESTS", "6000"))
MAX_GEMINI_REQUESTS = int(os.environ.get("MAX_GEMINI_REQUESTS", "1500"))
VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ---------------------------------------------------------------------------
# Request counters (thread-safe, persisted to disk so restarts don't reset)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_counter_dir = os.path.dirname(__file__)


def _counter_path(name: str) -> str:
    return os.path.join(_counter_dir, f".{name}_count")


def _read_count(name: str) -> int:
    try:
        with open(_counter_path(name)) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _write_count(name: str, n: int):
    with open(_counter_path(name), "w") as f:
        f.write(str(n))


def increment_and_check(name: str, max_requests: int) -> int:
    """Increment counter for `name`. Returns new count. Raises if over limit."""
    with _lock:
        count = _read_count(name) + 1
        if count > max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Workshop {name} API limit reached ({max_requests} requests). "
                       f"Budget cap hit — no more requests allowed.",
            )
        _write_count(name, count)
        return count


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="CV Workshop — Google Vision Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    vision_count = _read_count("vision")
    gemini_count = _read_count("gemini")
    return {
        "status": "ok",
        "vision": {
            "requests_used": vision_count,
            "requests_remaining": max(0, MAX_VISION_REQUESTS - vision_count),
            "max_requests": MAX_VISION_REQUESTS,
        },
        "gemini": {
            "requests_used": gemini_count,
            "requests_remaining": max(0, MAX_GEMINI_REQUESTS - gemini_count),
            "max_requests": MAX_GEMINI_REQUESTS,
        },
    }


@app.post("/vision")
async def vision(file: UploadFile = File(...)):
    """
    Proxy an image to Google Cloud Vision API.
    Returns labels, face annotations, object localizations, and safe-search.
    """
    if not VISION_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_VISION_API_KEY not set on the server.",
        )

    # Budget gate
    count = increment_and_check("vision", MAX_VISION_REQUESTS)

    # Read & validate image
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Encode to base64 for the Vision API
    b64_image = base64.b64encode(contents).decode("utf-8")

    # Build request — ask for multiple feature types in one call
    payload = {
        "requests": [
            {
                "image": {"content": b64_image},
                "features": [
                    {"type": "LABEL_DETECTION", "maxResults": 10},
                    {"type": "FACE_DETECTION", "maxResults": 5},
                    {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                    {"type": "SAFE_SEARCH_DETECTION"},
                    {"type": "IMAGE_PROPERTIES", "maxResults": 5},
                ],
            }
        ]
    }

    # Call Google Vision API
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{VISION_URL}?key={VISION_API_KEY}",
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Google Vision API error: {resp.text}",
        )

    data = resp.json()
    response = data.get("responses", [{}])[0]

    # Parse labels
    labels = [
        {"name": l["description"], "score": round(l["score"], 3)}
        for l in response.get("labelAnnotations", [])
    ]

    # Parse face annotations
    faces = []
    for f in response.get("faceAnnotations", []):
        faces.append({
            "joy": f.get("joyLikelihood", "UNKNOWN"),
            "sorrow": f.get("sorrowLikelihood", "UNKNOWN"),
            "anger": f.get("angerLikelihood", "UNKNOWN"),
            "surprise": f.get("surpriseLikelihood", "UNKNOWN"),
            "headwear": f.get("headwearLikelihood", "UNKNOWN"),
            "confidence": round(f.get("detectionConfidence", 0), 3),
            "boundingPoly": f.get("boundingPoly", {}),
        })

    # Parse object localization
    objects = [
        {
            "name": o["name"],
            "score": round(o["score"], 3),
            "boundingPoly": o.get("boundingPoly", {}),
        }
        for o in response.get("localizedObjectAnnotations", [])
    ]

    # Parse safe search
    safe_search = response.get("safeSearchAnnotation", {})

    # Parse dominant colors
    colors = []
    props = response.get("imagePropertiesAnnotation", {})
    for c in props.get("dominantColors", {}).get("colors", [])[:5]:
        rgb = c.get("color", {})
        colors.append({
            "r": int(rgb.get("red", 0)),
            "g": int(rgb.get("green", 0)),
            "b": int(rgb.get("blue", 0)),
            "score": round(c.get("score", 0), 3),
            "pixelFraction": round(c.get("pixelFraction", 0), 3),
        })

    return {
        "labels": labels,
        "faces": faces,
        "objects": objects,
        "safeSearch": safe_search,
        "dominantColors": colors,
        "requestsUsed": count,
        "requestsRemaining": max(0, MAX_VISION_REQUESTS - count),
    }


@app.post("/gemini")
async def gemini(file: UploadFile = File(...), prompt: str = "Describe this image in detail."):
    """
    Proxy an image + prompt to the Gemini API (multimodal).
    Returns the model's free-text narration of the image.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not set on the server.",
        )

    # Budget gate
    count = increment_and_check("gemini", MAX_GEMINI_REQUESTS)

    # Read & validate image
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        mime_type = f"image/{img.format.lower()}" if img.format else "image/jpeg"
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Encode to base64
    b64_image = base64.b64encode(contents).decode("utf-8")

    # Build Gemini request
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_image,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {resp.text}",
        )

    data = resp.json()

    # Extract text from response
    text = ""
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = "(Gemini returned no text)"

    return {
        "text": text,
        "prompt": prompt,
        "requestsUsed": count,
        "requestsRemaining": max(0, MAX_GEMINI_REQUESTS - count),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if not VISION_API_KEY:
        print("WARNING: GOOGLE_VISION_API_KEY is not set!")
        print("Set it with: export GOOGLE_VISION_API_KEY='your-key-here'")
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY is not set!")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
    print(f"Starting Vision + Gemini proxy on http://localhost:{port}")
    print(f"Vision request cap: {MAX_VISION_REQUESTS}")
    print(f"Gemini request cap: {MAX_GEMINI_REQUESTS}")
    uvicorn.run(app, host="0.0.0.0", port=port)
