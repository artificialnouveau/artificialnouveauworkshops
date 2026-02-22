"""
Workshop API Server — proxies requests to Replicate API.

Local:
    pip install flask flask-cors replicate
    export REPLICATE_API_TOKEN="r8_your_token_here"
    python server.py

Render:
    Set REPLICATE_API_TOKEN in Render environment variables.
    Start command: gunicorn server:app
"""

import json
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


@app.route("/api/txt2img", methods=["POST"])
def txt2img():
    body = request.get_json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    output = replicate.run(
        MODELS["txt2img"],
        input={"prompt": prompt, "num_outputs": 1, "output_format": "webp"},
    )
    images = [str(url) for url in output]
    return jsonify({"images": images})


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

    output = replicate.run(
        MODELS["img2img"],
        input={
            "prompt": prompt,
            "image": image,
            "prompt_strength": strength,
            "num_outputs": 1,
        },
    )
    images = [str(url) for url in output]
    return jsonify({"images": images})


@app.route("/api/img2txt", methods=["POST"])
def img2txt():
    body = request.get_json()
    image = body.get("image")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    output = replicate.run(
        MODELS["img2txt"],
        input={"image": image, "task": "image_captioning"},
    )
    caption = str(output)
    return jsonify({"caption": caption})


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

    # PhotoMaker requires "img" trigger word in the prompt
    if "img" not in prompt.lower():
        prompt = f"a photo of a person img, {prompt}"

    output = replicate.run(
        MODELS["photomaker"],
        input={
            "prompt": prompt,
            "input_image": image,
            "style_name": style,
            "num_outputs": min(num_outputs, 4),
        },
    )
    images = [str(url) for url in output]
    return jsonify({"images": images})


@app.route("/api/img3d", methods=["POST"])
def img3d():
    body = request.get_json()
    image = body.get("image")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    output = replicate.run(
        MODELS["img3d"],
        input={
            "image": image,
            "steps": 50,
            "guidance_scale": 5.5,
            "octree_resolution": 256,
            "remove_background": True,
        },
    )
    mesh_url = str(output.get("mesh", "")) if isinstance(output, dict) else str(output)
    return jsonify({"mesh": mesh_url})


@app.route("/api/txt3d", methods=["POST"])
def txt3d():
    body = request.get_json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    output = replicate.run(
        MODELS["txt3d"],
        input={
            "prompt": prompt,
            "batch_size": 1,
            "render_mode": "nerf",
            "render_size": 256,
            "guidance_scale": 15,
            "save_mesh": True,
        },
    )
    files = [str(url) for url in output]
    return jsonify({"files": files})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Workshop server running at: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=True)
