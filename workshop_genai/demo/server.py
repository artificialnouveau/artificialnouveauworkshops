"""
Workshop API Server — proxies requests to Replicate API.

Uses async predictions + polling to avoid Render's 30s request timeout.
The frontend calls /api/<model> to start a prediction, then polls
/api/prediction/<id> until it completes.

Local:
    pip install flask flask-cors replicate
    export REPLICATE_API_TOKEN="r8_your_token_here"
    python server.py

Render:
    Set REPLICATE_API_TOKEN in Render environment variables.
    Start command: gunicorn server:app
"""

import os

from flask import Flask, request, jsonify
from flask_cors import CORS

import replicate

app = Flask(__name__)
CORS(app)

# ── Replicate model identifiers ──────────────────────────────────────
MODELS = {
    "txt2img": "black-forest-labs/flux-schnell",
    "img2img": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "img2txt": "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
    "photomaker": "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
    "img3d": "tencent/hunyuan3d-2",
    "txt3d": "cjwbw/shap-e",
}


def start_prediction(model_key, model_input):
    """Start an async prediction and return its ID immediately."""
    model_ref = MODELS[model_key]
    if ":" in model_ref:
        # Versioned model — use version directly
        parts = model_ref.split(":")
        prediction = replicate.predictions.create(
            version=parts[1],
            input=model_input,
        )
    else:
        # Unversioned model — use model reference
        model = replicate.models.get(model_ref)
        version = model.latest_version
        prediction = replicate.predictions.create(
            version=version.id,
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

    pred_id = start_prediction("img2img", {
        "prompt": prompt, "image": image,
        "prompt_strength": strength, "num_outputs": 1,
    })
    return jsonify({"prediction_id": pred_id})


@app.route("/api/img2txt", methods=["POST"])
def img2txt():
    body = request.get_json()
    image = body.get("image")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    pred_id = start_prediction("img2txt", {
        "image": image, "task": "image_captioning",
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

    pred_id = start_prediction("photomaker", {
        "prompt": prompt, "input_image": image,
        "style_name": style, "num_outputs": min(num_outputs, 4),
    })
    return jsonify({"prediction_id": pred_id})


@app.route("/api/img3d", methods=["POST"])
def img3d():
    body = request.get_json()
    image = body.get("image")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    pred_id = start_prediction("img3d", {
        "image": image, "steps": 50, "guidance_scale": 5.5,
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
        "status": prediction.status,  # starting, processing, succeeded, failed, canceled
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
