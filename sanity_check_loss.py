"""
Sanity check for evaluate_model_masks and compute_mask_loss.

Tests: (1) model inference on images, (2) Dice + count penalty per image,
(3) composite loss under different weight combinations.
"""

import numpy as np
from pathlib import Path
import cv2

from surrogate_gp import evaluate_model_masks, compute_mask_loss, dice_loss, count_penalty

# ═══════════════════════════════════════════════════════════════════
# Paths — EDIT THESE
# ═══════════════════════════════════════════════════════════════════

# Path to a trained model file.
#   For unet / unet_modified  → .keras file saved by train_model_on_resolutions
#   For resnext               → .pth  file saved by train_model_on_resolutions
MODEL_PATH = "./models/clean_blackbox.keras"  # <-- set this

# Directory of clean (or synthetic) images with ground-truth masks.
#   Must contain paired image/mask files using the same convention as
#   load_image_mask_pairs: masks have "_mask" in the filename,
#   images do not.  PNG, JPG, or TIFF.
#IMAGE_DIR = "./prediction_noisy_images/"  # <-- set this
IMAGE_DIR = "./real_clean_images/"

# Temporary workspace (will be created/deleted each run).
PATCH_DIR = "./sanity_check_workspace"

# Model family — must match the saved model.
#   Options: "unet", "unet_modified", "resnext"
MODEL_FAMILY = "unet"

# Set to False to skip the matplotlib display.
SHOW_PLOTS = True

# ═══════════════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════════════
import tensorflow as tf
import segmenteverygrain as seg

if MODEL_FAMILY in {"unet", "unet_modified"}:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
elif MODEL_FAMILY == "resnext":
    import torch
    from segmenteverygrain.resnext_model import MaskingResNeXt
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MaskingResNeXt(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
else:
    raise ValueError(f"Unknown model_family: {MODEL_FAMILY}")

print(f"Loaded {MODEL_FAMILY} model from {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════════════
# Part 3 — Run model inference on all images
# ═══════════════════════════════════════════════════════════════════
print(f"\n--- evaluate_model_masks ---")
print(f"Image dir: {IMAGE_DIR}")
pred_probs, true_masks = evaluate_model_masks(
    model, IMAGE_DIR, PATCH_DIR, MODEL_FAMILY,
)
print(f"Processed {len(pred_probs)} image(s)")

# ═══════════════════════════════════════════════════════════════════
# Per-image breakdown: Dice + count
# ═══════════════════════════════════════════════════════════════════
print(f"\n--- Per-image loss components ---")
for i, (pred, true) in enumerate(zip(pred_probs, true_masks)):
    d = dice_loss([pred], [true])
    c = count_penalty([pred], [true])
    print(f"  Image {i}:  dice={d:.4f}  count_penalty={c:.4f}  pred_shape={pred.shape}  true_shape={true.shape}")

print(f"\n--- Overall averages ---")
d_avg = dice_loss(pred_probs, true_masks)
c_avg = count_penalty(pred_probs, true_masks)
print(f"  dice={d_avg:.4f}  count_penalty={c_avg:.4f}")

# ═══════════════════════════════════════════════════════════════════
# Part 4 — Sweep weight combinations
# ═══════════════════════════════════════════════════════════════════
print(f"\n--- Weight combinations (dice_weight, count_weight → composite) ---")
combos = [
    (1.0, 0.0),
    (1.0, 0.05),
    (1.0, 0.1),
    (1.0, 0.2),
    (1.0, 0.5),
    (1.0, 0.5996),
    (1.0, 1.0),
    (0.5, 0.1),
    (0.5, 0.5),
    (0.2, 1.0),
]
for dw, cw in combos:
    composite = compute_mask_loss(pred_probs, true_masks, dw, cw)
    print(f"  {dw:5.2f}  × dice   +  {cw:5.2f}  × count   =  {composite:.4f}")

# Suggested weight — aim for each term to contribute roughly equally.
if c_avg > 0:
    balanced_cw = 1.0 * (d_avg / c_avg)
    print(f"\nSuggested balanced count_weight ≈ {balanced_cw:.4f}  "
          f"(so 1.0 * dice ≈ count_weight * count_penalty)")

# ═══════════════════════════════════════════════════════════════════
# Visualization — repo-standard side-by-side display
# ═══════════════════════════════════════════════════════════════════
if SHOW_PLOTS:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from create_synthetic_images import load_image_mask_pairs

    # repo-standard colormap: black=background, steelblue=grain, orange=boundary
    cmap = ListedColormap(['black', 'steelblue', 'orange'])

    pairs = load_image_mask_pairs(IMAGE_DIR)

    for i, ((img_path, mask_path), pred, true) in enumerate(zip(pairs, pred_probs, true_masks)):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred_label = np.argmax(pred, axis=-1).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(img)
        axes[0].set_title("Input image")
        axes[0].axis("off")

        axes[1].imshow(true, cmap=cmap, vmin=0, vmax=2)
        axes[1].set_title("Ground truth")
        axes[1].axis("off")

        axes[2].imshow(pred_label, cmap=cmap, vmin=0, vmax=2)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        axes[3].imshow(img)
        axes[3].imshow(pred_label, cmap=cmap, vmin=0, vmax=2, alpha=0.4)
        axes[3].set_title("Overlay (prediction)")
        axes[3].axis("off")

        plt.suptitle(f"Image {i}  |  dice={dice_loss([pred], [true]):.4f}  "
                     f"count_pen={count_penalty([pred], [true]):.4f}")
        plt.tight_layout()
        plt.show()

# ═══════════════════════════════════════════════════════════════════
# Sanity: verify that a certain prediction scores 0 loss
# ═══════════════════════════════════════════════════════════════════
# A "certain" softmax: for each pixel, probability 1.0 for the true class, 0 elsewhere.
# dice_loss and count_penalty should both be 0.
print(f"\n--- Sanity: certain prediction → zero loss ---")
if len(true_masks) > 0:
    true_example = true_masks[0]
    h, w = true_example.shape
    certain = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        certain[:, :, c] = (true_example == c).astype(np.float32)

    d = dice_loss([certain], [true_example])
    c = count_penalty([certain], [true_example])
    print(f"  dice_loss(certain, true)       = {d:.6f}  (expected 0)")
    print(f"  count_penalty(certain, true)   = {c:.6f}  (expected 0)")
    assert d == 0, f"dice_loss should be 0, got {d}"
    assert c == 0, f"count_penalty should be 0, got {c}"
    print("  ✓  Both assertions passed.")
else:
    print("  No masks available — skipping.")
