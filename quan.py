"""
Compresses model checkpoints to FP16 (half precision) so they fit comfortably
under GitHub's 100MB limit — no external hosting needed.

Typical size reduction: ~50% (e.g. unet_best.pt 93MB -> ~47MB, vit_best.pt 82MB -> ~41MB)

Usage:
    Place this file inside your streamlit_app/ folder (same level as models/), then:
        python quantize.py

Run this ONCE on your original full-precision checkpoints. It creates new files
named *_fp16.pt alongside the originals. Your app's loaders already check for
these fp16 files first and fall back to full precision automatically — no other
code changes needed.
"""

import torch
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

FILES_TO_COMPRESS = [
    "unet_best.pt",
    "vit_best.pt",
    # add any other checkpoint here if it's also close to GitHub's 100MB limit, e.g.:
    # "yolov11_bccd_best.pt",
]


def compress_checkpoint(filename: str):
    src = MODEL_DIR / filename
    if not src.exists():
        print(f"  SKIP — {filename} not found in {MODEL_DIR}")
        return

    original_size_mb = src.stat().st_size / (1024 * 1024)

    state_dict = torch.load(src, map_location="cpu")

    # handle both raw state_dicts and checkpoints wrapped in a dict (e.g. {'model': ..., 'epoch': ...})
    if isinstance(state_dict, dict) and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        target_dict = state_dict
        wrapped = False
    else:
        # find the tensor-containing key (common patterns: 'state_dict', 'model_state_dict', 'model')
        wrapped = True
        tensor_key = next((k for k in ["state_dict", "model_state_dict", "model"] if k in state_dict), None)
        if tensor_key is None:
            print(f"  SKIP — {filename} has an unrecognized checkpoint structure, compress manually")
            return
        target_dict = state_dict[tensor_key]

    fp16_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in target_dict.items()}

    if wrapped:
        state_dict[tensor_key] = fp16_dict
        output = state_dict
    else:
        output = fp16_dict

    dest = MODEL_DIR / filename.replace(".pt", "_fp16.pt")
    torch.save(output, dest)

    new_size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  {filename}: {original_size_mb:.1f}MB -> {dest.name}: {new_size_mb:.1f}MB "
          f"({100 * (1 - new_size_mb/original_size_mb):.0f}% smaller)")


if __name__ == "__main__":
    print(f"Compressing checkpoints in {MODEL_DIR}...\n")
    for fname in FILES_TO_COMPRESS:
        compress_checkpoint(fname)
    print("\nDone. app.py already prefers *_fp16.pt files automatically — just push both "
          "the fp16 files and app.py to GitHub.")