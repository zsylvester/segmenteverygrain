"""
resnext_model.py
----------------
Masking ResNeXt: a pretrained ResNeXt50_32x4d encoder with a UNet-style
decoder head, adapted from:

  Satterlee et al. (2025) "Robust image-based cross-sectional grain boundary
  detection and characterization using machine learning"
  Journal of Intelligent Manufacturing 36:3067-3095
  https://doi.org/10.1007/s10845-024-02383-6

Architecture
------------
Encoder  : torchvision ResNeXt50_32x4d (pretrained on ImageNet)
           Skip connections taken from the stem and layers 1-4.

           For a 256×256 input the skip tensors have shapes:
               stem    128×128×64
               layer1   64× 64×256
               layer2   32× 32×512
               layer3   16× 16×1024
               layer4    8×  8×2048

Decoder  : 5 UpConv blocks + a 1×1 ConvOut layer.

           Each UpConv block:
               TransposeConv2d (same channels, stride 2) → BN + ReLU
               Conv2d 3×3 (reduces to target channels)   → BN + ReLU
               Conv2d 1×1 (maintains channels)           → BN + ReLU
           followed by concatenation of the matching encoder skip tensor.

           Channel flow (256×256 input):
               layer4  →  UpConv1 (→1024)  + layer3  →  2048
               2048    →  UpConv2 (→ 512)  + layer2  →  1024
               1024    →  UpConv3 (→ 256)  + layer1  →   512
                512    →  UpConv4 (→  64)  + stem    →   128
                128    →  UpConv5 (→  32)  no skip   →    32
                 32    →  ConvOut (→   3)  softmax   →     3

Output   : (N, 3, H, W) softmax probabilities, matching the existing UNet
           convention: channel 0 = background, 1 = grain interior,
           2 = grain boundary.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


# ── building blocks ──────────────────────────────────────────────────────────

class UpConvBlock(nn.Module):
    """
    Single decoder block:
      TransposeConv2d (same ch, stride=2)  → BN + ReLU
      Conv2d 3×3      (in_ch → out_ch)     → BN + ReLU
      Conv2d 1×1      (out_ch → out_ch)    → BN + ReLU
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── full model ────────────────────────────────────────────────────────────────

class MaskingResNeXt(nn.Module):
    """
    ResNeXt50_32x4d encoder with a UNet decoder head.

    Parameters
    ----------
    num_classes : int
        Number of output channels (default 3: background / interior / boundary).
    pretrained : bool
        Whether to initialise the encoder with ImageNet weights (default True).
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()

        weights = tvm.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnext50_32x4d(weights=weights)

        # ── encoder (keep all layers, extract skip tensors in forward) ────────
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool   = backbone.maxpool
        self.layer1 = backbone.layer1   # 64ch→256ch
        self.layer2 = backbone.layer2   # 256→512
        self.layer3 = backbone.layer3   # 512→1024
        self.layer4 = backbone.layer4   # 1024→2048

        # ── decoder ───────────────────────────────────────────────────────────
        # Each UpConv output_ch == corresponding skip_ch, so after concat the
        # channel count doubles cleanly.
        #
        #   block    in_ch   out_ch   skip_ch   post-concat
        #   UpConv1  2048    1024     1024      2048
        #   UpConv2  2048     512      512      1024
        #   UpConv3  1024     256      256       512
        #   UpConv4   512      64       64       128
        #   UpConv5   128      32        –        32
        self.up1 = UpConvBlock(2048, 1024)
        self.up2 = UpConvBlock(2048,  512)
        self.up3 = UpConvBlock(1024,  256)
        self.up4 = UpConvBlock( 512,   64)
        self.up5 = UpConvBlock( 128,   32)

        self.conv_out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── encode ────────────────────────────────────────────────────────────
        s  = self.stem(x)          # H/2  ×  W/2  ×  64   (skip)
        x  = self.pool(s)          # H/4  ×  W/4  ×  64
        l1 = self.layer1(x)        # H/4  ×  W/4  × 256   (skip)
        l2 = self.layer2(l1)       # H/8  ×  W/8  × 512   (skip)
        l3 = self.layer3(l2)       # H/16 × W/16  ×1024   (skip)
        l4 = self.layer4(l3)       # H/32 × W/32  ×2048

        # ── decode ────────────────────────────────────────────────────────────
        x = torch.cat([self.up1(l4), l3], dim=1)   # →  H/16 ×1024+1024=2048
        x = torch.cat([self.up2(x),  l2], dim=1)   # →  H/8  × 512+ 512=1024
        x = torch.cat([self.up3(x),  l1], dim=1)   # →  H/4  × 256+ 256= 512
        x = torch.cat([self.up4(x),  s],  dim=1)   # →  H/2  ×  64+  64= 128
        x = self.up5(x)                             # →  H    ×  32
        x = self.conv_out(x)                        # →  H    ×  num_classes

        return torch.softmax(x, dim=1)


# ── loss ──────────────────────────────────────────────────────────────────────

def weighted_crossentropy_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: tuple[float, ...] = (0.6, 1.0, 5.0),
    device: torch.device = None,
) -> torch.Tensor:
    """
    Weighted cross-entropy loss matching the Keras UNet's loss function.

    Parameters
    ----------
    pred : (N, C, H, W) softmax probabilities
    target : (N, H, W) integer class labels  (0 / 1 / 2)
    weights : per-class weights (background, interior, boundary)
    """
    if device is None:
        device = pred.device
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)(pred, target.long())


# ── convenience ───────────────────────────────────────────────────────────────

def load_resnext(path: str, num_classes: int = 3, device: str = "cpu") -> MaskingResNeXt:
    """Load a saved MaskingResNeXt checkpoint."""
    model = MaskingResNeXt(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_patch_torch(
    model: MaskingResNeXt,
    image_patch: "np.ndarray",
    device: str = "cpu",
) -> "np.ndarray":
    """
    Run inference on a single HxWx3 uint8 numpy patch.
    Returns an HxW×3 float32 softmax probability array, matching the shape
    that ``seg.predict_image`` produces from the Keras UNet.
    """
    import numpy as np
    import torch

    x = torch.from_numpy(image_patch.astype("float32") / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)  # 1×3×H×W
    with torch.no_grad():
        out = model(x)                               # 1×3×H×W
    return out.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H×W×3
