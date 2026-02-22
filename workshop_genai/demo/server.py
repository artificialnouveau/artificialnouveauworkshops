"""
Workshop API Server — proxies requests to Replicate API.

Uses async predictions + polling to avoid Render's 30s request timeout.
The frontend calls /api/<model> to start a prediction, then polls
/api/prediction/<id> until it completes.

Images are uploaded to Replicate's file service first (fast), then
the file URL is passed to the prediction (no huge base64 in JSON).

Local:
    pip install flask flask-cors replicate
    export REPLICATE_API_TOKEN="r8_your_token_here"
    python server.py

Render:
    Set REPLICATE_API_TOKEN in Render environment variables.
    Start command: gunicorn server:app --timeout 120 --workers 2
"""

import base64
import io
import os

from flask import Flask, request, jsonify
from flask_cors import CORS

import replicate

app = Flask(__name__)
CORS(app)

# ── Replicate model identifiers ──────────────────────────────────────
MODELS = {
    "txt2img": "black-forest-labs/flux-schnell",
    "img2img": "bxclib2/flux_img2img",
    "img2txt": "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
    "photomaker": "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
    "img3d": "tencent/hunyuan3d-2:b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb",
    "txt3d": "cjwbw/shap-e:5957069d5c509126a73c7cb68abcddbb985aeefa4d318e7c63ec1352ce6da68c",
}


def upload_data_uri(data_uri):
    """Convert a data URI to a Replicate file upload URL."""
    # Parse data URI: data:image/jpeg;base64,/9j/4AAQ...
    header, b64data = data_uri.split(",", 1)
    mime = header.split(":")[1].split(";")[0]
    ext = mime.split("/")[1].replace("jpeg", "jpg")
    raw = base64.b64decode(b64data)
    file_obj = io.BytesIO(raw)
    file_obj.name = f"upload.{ext}"
    uploaded = replicate.files.create(file_obj, content_type=mime)
    return uploaded.urls["get"]


def start_prediction(model_key, model_input):
    """Start an async prediction and return its ID immediately."""
    model_ref = MODELS[model_key]
    if ":" in model_ref:
        # Versioned model (community) — use version hash
        version = model_ref.split(":")[1]
        prediction = replicate.predictions.create(
            version=version,
            input=model_input,
        )
    else:
        # Official model (no version) — use model parameter
        prediction = replicate.predictions.create(
            model=model_ref,
            input=model_input,
        )
    return prediction.id


# ── Start endpoints (return prediction ID immediately) ───────────────

@app.route("/api/txt2img", methods=["POST"])
def txt2img():
    body = request.get_json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    pred_id = start_prediction("txt2img", {
        "prompt": prompt, "num_outputs": 1, "output_format": "webp",
    })
    return jsonify({"prediction_id": pred_id})


@app.route("/api/img2img", methods=["POST"])
def img2img():
    body = request.get_json()
    prompt = body.get("prompt", "").strip()
    image = body.get("image")
    strength = body.get("strength", 0.7)

    if not image:
        return jsonify({"error": "Image is required"}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    image_url = upload_data_uri(image)
    pred_id = start_prediction("img2img", {
        "positive_prompt": prompt, "image": image_url,
        "denoising": strength,
    })
    return jsonify({"prediction_id": pred_id})


@app.route("/api/img2txt", methods=["POST"])
def img2txt():
    body = request.get_json()
    image = body.get("image")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    image_url = upload_data_uri(image)
    pred_id = start_prediction("img2txt", {
        "image": image_url, "task": "image_captioning",
    })
    return jsonify({"prediction_id": pred_id})


@app.route("/api/photomaker", methods=["POST"])
def photomaker():
    body = request.get_json()
    prompt = body.get("prompt", "").strip()
    image = body.get("image")
    style = body.get("style", "(No style)")
    num_outputs = body.get("num_outputs", 2)

    if not image:
        return jsonify({"error": "Face image is required"}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if "img" not in prompt.lower():
        prompt = f"a photo of a person img, {prompt}"

    image_url = upload_data_uri(image)
    pred_id = start_prediction("photomaker", {
        "prompt": prompt, "input_image": image_url,
        "style_name": style, "num_outputs": min(num_outputs, 4),
    })
    return jsonify({"prediction_id": pred_id})


@app.route("/api/img3d", methods=["POST"])
def img3d():
    body = request.get_json()
    image = body.get("image")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    image_url = upload_data_uri(image)
    pred_id = start_prediction("img3d", {
        "image": image_url, "steps": 50, "guidance_scale": 5.5,
        "octree_resolution": 256, "remove_background": True,
    })
    return jsonify({"prediction_id": pred_id})


@app.route("/api/txt3d", methods=["POST"])
def txt3d():
    body = request.get_json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    pred_id = start_prediction("txt3d", {
        "prompt": prompt, "batch_size": 1, "render_mode": "nerf",
        "render_size": 256, "guidance_scale": 15, "save_mesh": True,
    })
    return jsonify({"prediction_id": pred_id})


# ── Poll endpoint (frontend checks this every 2s) ────────────────────

@app.route("/api/prediction/<prediction_id>")
def get_prediction(prediction_id):
    prediction = replicate.predictions.get(prediction_id)

    result = {
        "status": prediction.status,
    }

    if prediction.status == "succeeded":
        result["output"] = prediction.output
    elif prediction.status == "failed":
        result["error"] = prediction.error or "Prediction failed"

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Workshop server running at: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=True)
