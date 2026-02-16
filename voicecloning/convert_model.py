#!/usr/bin/env python3
"""
convert_model.py — Convert an RVC .pth voice model to .onnx for browser use.

Usage:
    python convert_model.py --pth my_voice.pth --out my_voice.onnx

Requirements:
    pip install torch onnx

This script extracts the net_g (generator) from an RVC .pth checkpoint
and exports it as an ONNX model compatible with ONNX Runtime Web.

The exported model expects these inputs:
  - phone:         [1, T, 768]  float32  (ContentVec features)
  - phone_lengths: [1]          int64    (sequence length)
  - pitch:         [1, T]       int64    (coarse pitch, 0-255)
  - pitchf:        [1, T]       float32  (F0 in Hz)
  - ds:            [1]          int64    (speaker ID, usually 0)
  - rnd:           [1, 192, T]  float32  (random noise)
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Convert RVC .pth to .onnx")
    parser.add_argument("--pth", required=True, help="Path to RVC .pth model file")
    parser.add_argument("--out", required=True, help="Output .onnx file path")
    parser.add_argument("--vec-channels", type=int, default=768,
                        help="Feature dimension (256 for RVC v1, 768 for RVC v2)")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    args = parser.parse_args()

    if not os.path.isfile(args.pth):
        print(f"Error: File not found: {args.pth}")
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print(f"Loading checkpoint: {args.pth}")
    checkpoint = torch.load(args.pth, map_location="cpu", weights_only=False)

    # RVC checkpoints may store the model under different keys
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    config = checkpoint.get("config", None)
    if config:
        print(f"Model config: {config}")

    # Try to use RVC's own model classes if available
    try:
        # Attempt to import from RVC project (if installed or in path)
        sys.path.insert(0, os.path.dirname(args.pth))
        from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
        print("Using RVC's built-in ONNX model class.")
        use_rvc_class = True
    except ImportError:
        print("RVC model classes not found. Using generic export approach.")
        use_rvc_class = False

    if use_rvc_class and config:
        # Build model from RVC config
        model = SynthesizerTrnMsNSFsidM(*config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        # Generic approach: wrap state dict in a minimal forward module
        print("Creating minimal wrapper for ONNX export...")
        print("Note: For best results, run this from within the RVC project directory.")
        print()
        print("Alternative: Use the RVC WebUI directly:")
        print("  1. Open RVC WebUI")
        print("  2. Go to the 'Export ONNX' tab")
        print("  3. Select your .pth file")
        print("  4. Click 'Export'")
        print("  5. Upload the resulting .onnx file to the browser app")
        print()

        # Try a direct torch.jit approach
        class MinimalWrapper(torch.nn.Module):
            def __init__(self, sd):
                super().__init__()
                # Store parameters
                for k, v in sd.items():
                    safe_k = k.replace(".", "_")
                    if isinstance(v, torch.Tensor):
                        self.register_buffer(safe_k, v)

            def forward(self, phone, phone_lengths, pitch, pitchf, ds, rnd):
                # Placeholder — real inference requires the full model architecture
                # This export is for shape/interface verification only
                T = phone.shape[1]
                return rnd.new_zeros(1, 1, T * 256)

        model = MinimalWrapper(state_dict)
        model.eval()
        print("WARNING: Using placeholder model. For real voice conversion,")
        print("export from RVC WebUI or ensure RVC source code is in your Python path.")

    # Create dummy inputs
    T = 100  # dummy sequence length
    dummy_phone = torch.randn(1, T, args.vec_channels)
    dummy_phone_lengths = torch.tensor([T], dtype=torch.long)
    dummy_pitch = torch.randint(0, 256, (1, T), dtype=torch.long)
    dummy_pitchf = torch.randn(1, T)
    dummy_ds = torch.tensor([0], dtype=torch.long)
    dummy_rnd = torch.randn(1, 192, T)

    print(f"Exporting to ONNX (opset {args.opset})...")
    try:
        torch.onnx.export(
            model,
            (dummy_phone, dummy_phone_lengths, dummy_pitch, dummy_pitchf, dummy_ds, dummy_rnd),
            args.out,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
            output_names=["audio"],
            dynamic_axes={
                "phone": {1: "seq_len"},
                "pitch": {1: "seq_len"},
                "pitchf": {1: "seq_len"},
                "rnd": {2: "seq_len"},
                "audio": {2: "audio_len"},
            },
            opset_version=args.opset,
        )
        size_mb = os.path.getsize(args.out) / (1024 * 1024)
        print(f"Exported: {args.out} ({size_mb:.1f} MB)")
        print("Upload this .onnx file to the browser voice cloning app.")
    except Exception as e:
        print(f"Export failed: {e}")
        print()
        print("Recommended approach:")
        print("  1. Clone the RVC repo: git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI")
        print("  2. Place your .pth file in the RVC directory")
        print("  3. Use the 'Export ONNX' tab in the WebUI")
        sys.exit(1)

if __name__ == "__main__":
    main()
