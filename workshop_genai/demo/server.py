"""
Workshop Proxy Server — serves static files + proxies to Replicate API.

Setup:
    pip install replicate

Run:
    export REPLICATE_API_TOKEN="r8_your_token_here"
    python server.py

Then open http://localhost:8000 in your browser.
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

try:
    import replicate
except ImportError:
    print("Error: 'replicate' package not found. Install it with:")
    print("  pip install replicate")
    sys.exit(1)

# Check for API token
if not os.environ.get("REPLICATE_API_TOKEN"):
    print("Error: REPLICATE_API_TOKEN environment variable not set.")
    print("  export REPLICATE_API_TOKEN='r8_your_token_here'")
    print("  Get your token at: https://replicate.com/account/api-tokens")
    sys.exit(1)


# ── Replicate model identifiers ──────────────────────────────────────
MODELS = {
    "txt2img": "black-forest-labs/flux-schnell",
    "img2img": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "img2txt": "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
    "photomaker": "tencentarc/photomaker:ddfc2b08d209f9fa8c1eca692712918bd449f695dabb4a958da31802a9570fe4",
    "img3d": "tencent/hunyuan3d-2",
    "txt3d": "cjwbw/shap-e",
}


class WorkshopHandler(SimpleHTTPRequestHandler):
    """Serves static files and proxies API requests to Replicate."""

    def end_headers(self):
        # CORS headers for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)

        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Invalid JSON"})
            return

        try:
            if self.path == "/api/txt2img":
                result = self.handle_txt2img(body)
            elif self.path == "/api/img2img":
                result = self.handle_img2img(body)
            elif self.path == "/api/img2txt":
                result = self.handle_img2txt(body)
            elif self.path == "/api/photomaker":
                result = self.handle_photomaker(body)
            elif self.path == "/api/img3d":
                result = self.handle_img3d(body)
            elif self.path == "/api/txt3d":
                result = self.handle_txt3d(body)
            else:
                self.send_json(404, {"error": "Unknown endpoint"})
                return

            self.send_json(200, result)

        except Exception as e:
            print(f"Error on {self.path}: {e}")
            self.send_json(500, {"error": str(e)})

    def send_json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    # ── Handlers ──────────────────────────────────────────────────────

    def handle_txt2img(self, body):
        prompt = body.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Prompt is required")

        output = replicate.run(
            MODELS["txt2img"],
            input={
                "prompt": prompt,
                "num_outputs": 1,
                "output_format": "webp",
            },
        )
        # flux-schnell returns a list of FileOutput objects
        images = [str(url) for url in output]
        return {"images": images}

    def handle_img2img(self, body):
        prompt = body.get("prompt", "").strip()
        image = body.get("image")  # data URI from browser
        strength = body.get("strength", 0.7)

        if not image:
            raise ValueError("Image is required")
        if not prompt:
            raise ValueError("Prompt is required")

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
        return {"images": images}

    def handle_img2txt(self, body):
        image = body.get("image")  # data URI from browser
        if not image:
            raise ValueError("Image is required")

        output = replicate.run(
            MODELS["img2txt"],
            input={
                "image": image,
                "task": "image_captioning",
            },
        )
        caption = str(output)
        return {"caption": caption}

    def handle_photomaker(self, body):
        prompt = body.get("prompt", "").strip()
        image = body.get("image")  # data URI from browser
        style = body.get("style", "(No style)")
        num_outputs = body.get("num_outputs", 2)

        if not image:
            raise ValueError("Face image is required")
        if not prompt:
            raise ValueError("Prompt is required")

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
        return {"images": images}

    def handle_img3d(self, body):
        image = body.get("image")  # data URI from browser
        if not image:
            raise ValueError("Image is required")

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
        # Returns {"mesh": "https://...glb"}
        mesh_url = str(output.get("mesh", "")) if isinstance(output, dict) else str(output)
        return {"mesh": mesh_url}

    def handle_txt3d(self, body):
        prompt = body.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Prompt is required")

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
        # Returns a list of URIs (GIF previews and/or mesh files)
        files = [str(url) for url in output]
        return {"files": files}


def main():
    port = int(os.environ.get("PORT", 8000))

    print(f"\n  Workshop server running at: http://localhost:{port}")
    print(f"  Replicate token: ...{os.environ['REPLICATE_API_TOKEN'][-4:]}")
    print(f"  Press Ctrl+C to stop\n")

    server = HTTPServer(("", port), WorkshopHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
