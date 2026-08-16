"""
INT8 quantization — pushes past fp16 to get ViT and U-Net under GitHub's 25MB
web-uploader limit.

Two different methods are used because the layer types differ:
  - ViT is almost entirely nn.Linear -> DYNAMIC quantization (safe, no calibration
    data needed, minimal accuracy impact).
  - U-Net is almost entirely nn.Conv2d -> dynamic quantization does NOT touch
    Conv2d at all, so we use STATIC quantization instead, which requires running
    a few calibration batches through the model first. This script uses random
    noise as a calibration stand-in since you don't have a labeled set handy —
    that's a real compromise: re-check your Dice/IoU score on this quantized
    U-Net afterward (see the note printed at the end) before trusting it fully.

Usage:
    python quantize_int8.py

Reads from ./models/*_fp16.pt (or falls back to the original *.pt), writes
./models/*_int8.pt. Your app's loaders need one small addition to try these
files first — see the loader snippet printed at the end of this script's output.
"""

import torch
import torch.nn as nn
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"


def get_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# ============================================================
# ViT — dynamic quantization (Linear layers)
# ============================================================

def quantize_vit():
    src = MODEL_DIR / "vit_best_fp16.pt"
    if not src.exists():
        src = MODEL_DIR / "vit_best.pt"
    if not src.exists():
        print("  SKIP ViT — no checkpoint found")
        return

    try:
        import timm
    except ImportError:
        print("  SKIP ViT — `timm` not installed (pip install timm)")
        return

    CLASSIFICATION_CLASSES = ['neutrophil', 'lymphocyte', 'monocyte', 'eosinophil', 'basophil']

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=len(CLASSIFICATION_CLASSES))
    state_dict = torch.load(src, map_location="cpu")
    state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    dest = MODEL_DIR / "vit_best_int8.pt"
    torch.save(quantized.state_dict(), dest)

    print(f"  ViT: {get_size_mb(src):.1f}MB -> {dest.name}: {get_size_mb(dest):.1f}MB")


# ============================================================
# U-Net — static quantization (Conv2d layers, needs calibration)
# ============================================================

def quantize_unet(n_calibration_batches: int = 20):
    src = MODEL_DIR / "unet_best_fp16.pt"
    if not src.exists():
        src = MODEL_DIR / "unet_best.pt"
    if not src.exists():
        print("  SKIP U-Net — no checkpoint found")
        return

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("  SKIP U-Net — `segmentation-models-pytorch` not installed")
        return

    SEGMENTATION_CLASSES = ['background', 'cytoplasm', 'nucleus']

    model_fp32 = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                           in_channels=3, classes=len(SEGMENTATION_CLASSES))
    state_dict = torch.load(src, map_location="cpu")
    state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
    model_fp32.load_state_dict(state_dict)
    model_fp32.eval()

    # Static quantization setup
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

    # Calibration pass — using random noise as a stand-in. This is a real
    # compromise: for a properly validated result, replace this with a loop
    # over ~20-50 real images from your test set instead.
    print(f"  Calibrating U-Net with {n_calibration_batches} random batches "
          f"(replace with real images for a properly validated result)...")
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            dummy_input = torch.randn(1, 3, 224, 224)
            model_prepared(dummy_input)

    model_int8 = torch.quantization.convert(model_prepared, inplace=False)

    dest = MODEL_DIR / "unet_best_int8.pt"
    torch.save(model_int8.state_dict(), dest)

    print(f"  U-Net: {get_size_mb(src):.1f}MB -> {dest.name}: {get_size_mb(dest):.1f}MB")


if __name__ == "__main__":
    print(f"INT8-quantizing checkpoints in {MODEL_DIR}...\n")
    print("ViT (dynamic quantization):")
    quantize_vit()
    print("\nU-Net (static quantization, calibrated on random noise):")
    quantize_unet()

    print("\n" + "="*70)
    print("IMPORTANT — loading these files requires different code than fp16:")
    print("="*70)
    print("""
Quantized models are not just a state_dict swap — the model architecture
itself must be converted to its quantized form BEFORE loading the weights,
or PyTorch won't know how to map the int8 tensors back in. In app.py, replace
the CNN/U-Net loader bodies with something like:

    # ViT
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=5)
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(torch.load('models/vit_best_int8.pt', map_location='cpu'))
    model.eval()

    # U-Net
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model, inplace=False)
    model = torch.quantization.convert(model_prepared, inplace=False)
    model.load_state_dict(torch.load('models/unet_best_int8.pt', map_location='cpu'))
    model.eval()

IMPORTANT CAVEAT: the U-Net calibration above used random noise, not real
smear images, because none were available at quantization time. Static
quantization's accuracy is sensitive to calibration data matching real input
statistics. Before trusting this int8 U-Net, re-run your Model 3 test-set
Dice/IoU evaluation against it and compare to your original ~0.971 Dice score.
If it's noticeably worse, re-run quantize_unet() with real calibration images
instead of torch.randn() — swap the dummy_input line for a DataLoader over a
handful of real test images (no labels needed, just the images).
""")