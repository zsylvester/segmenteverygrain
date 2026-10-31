"""
PyTorch implementation of the segmenteverygrain U-Net.

This module provides a pure-PyTorch version of the U-Net that was originally
defined with Keras in :func:`segmenteverygrain.Unet`, together with helpers to
load and save models and to convert existing Keras ``.keras`` model files into
the PyTorch format.

The architecture is identical to the Keras model: a five-level encoder with
16/32/64/128/256 filters, each level consisting of two 3x3 convolutions with
ReLU activation followed by batch normalization, 2x2 max pooling between
levels, and a symmetric decoder using 3x3 stride-2 transposed convolutions and
skip connections. The output layer is a 1x1 convolution that returns logits for
three classes (background, grain, grain boundary).

Notes
-----
Keras ``Conv2DTranspose(kernel_size=3, strides=2, padding="same")`` pads
asymmetrically in TensorFlow: the full-size transposed convolution output is
cropped by one pixel on the bottom and right only. :class:`_UpConv` reproduces
that behavior so that converted Keras weights give the same output in PyTorch.
The naive translation (``ConvTranspose2d(padding=1, output_padding=1)``)
shifts the result by one pixel and is *not* equivalent.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default per-channel class weights used by the weighted cross-entropy loss
# (background, grain, boundary). Kept here for reference by the training code.
DEFAULT_CLASS_WEIGHTS = (0.6, 1.0, 5.0)

_FORMAT_VERSION = 1


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolve the compute device, auto-detecting a GPU when none is specified.

    Parameters
    ----------
    device : str or torch.device, optional
        Explicit device (``"cuda"``, ``"mps"``, ``"cpu"``, ...). When None,
        CUDA is used if available, then Apple-Silicon MPS, otherwise CPU.

    Returns
    -------
    torch.device
        The resolved device.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _ConvBlock(nn.Module):
    """Two 3x3 convolutions with ReLU, followed by batch normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Keras BatchNormalization defaults: epsilon=1e-3, momentum=0.99.
        # PyTorch's momentum is defined as (1 - Keras momentum).
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.bn(x)


class _UpConv(nn.Module):
    """
    3x3 stride-2 transposed convolution matching Keras ``padding="same"``.

    The full transposed-convolution output has size ``2*H + 1``; TensorFlow
    crops one row/column from the bottom/right to obtain ``2*H``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, :-1, :-1]


class UNet(nn.Module):
    """
    U-Net for semantic segmentation of grain images.

    Parameters
    ----------
    in_channels : int
        Number of input image channels. Defaults to 3 (RGB).
    num_classes : int
        Number of output classes. Defaults to 3 (background, grain, boundary).
    base_filters : int
        Number of filters in the first encoder level; each deeper level
        doubles it. Defaults to 16.

    Notes
    -----
    ``forward`` expects a float tensor of shape ``(N, C, H, W)`` with values in
    ``[0, 1]`` and ``H``, ``W`` divisible by 16. It returns raw logits of shape
    ``(N, num_classes, H, W)``; apply ``softmax`` over dimension 1 to obtain
    class probabilities. Use :func:`predict_batch` for a numpy, channels-last
    convenience wrapper.
    """

    def __init__(
        self, in_channels: int = 3, num_classes: int = 3, base_filters: int = 16
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        f = [base_filters * 2**i for i in range(5)]  # 16, 32, 64, 128, 256

        self.enc1 = _ConvBlock(in_channels, f[0])
        self.enc2 = _ConvBlock(f[0], f[1])
        self.enc3 = _ConvBlock(f[1], f[2])
        self.enc4 = _ConvBlock(f[2], f[3])
        self.enc5 = _ConvBlock(f[3], f[4])

        self.up6 = _UpConv(f[4], f[3])
        self.dec6 = _ConvBlock(f[3] * 2, f[3])
        self.up7 = _UpConv(f[3], f[2])
        self.dec7 = _ConvBlock(f[2] * 2, f[2])
        self.up8 = _UpConv(f[2], f[1])
        self.dec8 = _ConvBlock(f[1] * 2, f[1])
        self.up9 = _UpConv(f[1], f[0])
        self.dec9 = _ConvBlock(f[0] * 2, f[0])

        self.out = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(F.max_pool2d(c1, 2))
        c3 = self.enc3(F.max_pool2d(c2, 2))
        c4 = self.enc4(F.max_pool2d(c3, 2))
        c5 = self.enc5(F.max_pool2d(c4, 2))

        d6 = self.dec6(torch.cat([self.up6(c5), c4], dim=1))
        d7 = self.dec7(torch.cat([self.up7(d6), c3], dim=1))
        d8 = self.dec8(torch.cat([self.up8(d7), c2], dim=1))
        d9 = self.dec9(torch.cat([self.up9(d8), c1], dim=1))
        return self.out(d9)

    @property
    def config(self) -> dict:
        """Constructor arguments needed to rebuild this model."""
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_filters": self.base_filters,
        }


def save_unet(model: UNet, path: str) -> None:
    """
    Save a :class:`UNet` (weights and architecture configuration) to a file.

    Parameters
    ----------
    model : UNet
        The model to save.
    path : str
        Destination path; ``.pt`` is the conventional extension.
    """
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "format_version": _FORMAT_VERSION,
            "config": model.config,
            "state_dict": state_dict,
        },
        path,
    )


def load_unet(
    path: str, device: Optional[Union[str, torch.device]] = None
) -> UNet:
    """
    Load a :class:`UNet` saved with :func:`save_unet`.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` file.
    device : str or torch.device, optional
        Device to place the model on. When None, a GPU (CUDA or MPS) is used
        if available, otherwise CPU.

    Returns
    -------
    UNet
        The model in evaluation mode on the requested device.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"U-Net model file not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{path} is not a segmenteverygrain U-Net file. Keras '.keras' models "
            "must first be converted with segmenteverygrain.convert_keras_model()."
        )
    model = UNet(**checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(get_device(device))
    model.eval()
    return model


@torch.no_grad()
def predict_batch(
    model: UNet, tiles: np.ndarray, batch_size: int = 32
) -> np.ndarray:
    """
    Run the U-Net on a batch of image tiles and return class probabilities.

    Parameters
    ----------
    model : UNet
        A model in evaluation mode (as returned by :func:`load_unet`).
    tiles : numpy.ndarray
        Array of shape ``(N, H, W, 3)`` (or ``(H, W, 3)`` for a single tile)
        with pixel values scaled to ``[0, 1]``.
    batch_size : int
        Number of tiles per forward pass. Defaults to 32.

    Returns
    -------
    numpy.ndarray
        Softmax probabilities of shape ``(N, H, W, num_classes)`` (or
        ``(H, W, num_classes)`` for a single tile), float32.
    """
    single = tiles.ndim == 3
    if single:
        tiles = tiles[None]
    if tiles.ndim != 4 or tiles.shape[-1] != model.in_channels:
        raise ValueError(
            f"tiles must have shape (N, H, W, {model.in_channels}); got {tiles.shape}"
        )
    device = next(model.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(tiles, dtype=np.float32))
    x = x.permute(0, 3, 1, 2)
    outputs = []
    for start in range(0, x.shape[0], batch_size):
        logits = model(x[start : start + batch_size].to(device))
        probs = torch.softmax(logits, dim=1).permute(0, 2, 3, 1)
        outputs.append(probs.cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    return result[0] if single else result


def _keras_to_torch_conv(weight: np.ndarray) -> torch.Tensor:
    """Keras conv kernel ``(kh, kw, in, out)`` -> PyTorch ``(out, in, kh, kw)``.

    The same transpose maps a Keras ``Conv2DTranspose`` kernel
    ``(kh, kw, out, in)`` to PyTorch ``ConvTranspose2d`` ``(in, out, kh, kw)``.
    """
    return torch.from_numpy(np.ascontiguousarray(np.transpose(weight, (3, 2, 0, 1))))


def convert_keras_model(
    keras_path: str, pt_path: Optional[str] = None, verify: bool = True
) -> UNet:
    """
    Convert a Keras ``.keras`` U-Net (created with :func:`segmenteverygrain.Unet`)
    into a PyTorch :class:`UNet`.

    Requires the ``keras`` package (and a Keras backend such as TensorFlow)
    to be installed, as the Keras file has to be read with Keras itself.

    Parameters
    ----------
    keras_path : str
        Path to the ``.keras`` model file.
    pt_path : str, optional
        If given, the converted model is saved to this path with
        :func:`save_unet`.
    verify : bool
        When True (default), run a random input through both models and raise
        if the outputs differ by more than ``1e-3``.

    Returns
    -------
    UNet
        The converted model in evaluation mode, on the CPU.
    """
    try:
        import keras
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Converting a Keras model requires the 'keras' package and a Keras "
            "backend (e.g. 'pip install tensorflow'). "
        ) from exc

    keras_model = keras.saving.load_model(keras_path, compile=False)

    layers_by_type: dict[str, list] = {"Conv2D": [], "BatchNormalization": [], "Conv2DTranspose": []}
    for layer in keras_model.layers:
        name = type(layer).__name__
        if name in layers_by_type:
            layers_by_type[name].append(layer)
    convs = layers_by_type["Conv2D"]
    bns = layers_by_type["BatchNormalization"]
    ups = layers_by_type["Conv2DTranspose"]

    if len(convs) != 19 or len(bns) != 9 or len(ups) != 4:
        raise ValueError(
            f"{keras_path} does not have the expected segmenteverygrain U-Net "
            f"architecture (found {len(convs)} Conv2D, {len(bns)} BatchNormalization, "
            f"{len(ups)} Conv2DTranspose layers)."
        )

    in_channels = int(convs[0].get_weights()[0].shape[2])
    base_filters = int(convs[0].get_weights()[0].shape[3])
    num_classes = int(convs[-1].get_weights()[0].shape[3])
    model = UNet(in_channels=in_channels, num_classes=num_classes, base_filters=base_filters)

    blocks = [
        model.enc1, model.enc2, model.enc3, model.enc4, model.enc5,
        model.dec6, model.dec7, model.dec8, model.dec9,
    ]
    with torch.no_grad():
        conv_idx = 0
        for block, bn in zip(blocks, bns):
            for torch_conv in (block.conv1, block.conv2):
                w, b = convs[conv_idx].get_weights()
                torch_conv.weight.copy_(_keras_to_torch_conv(w))
                torch_conv.bias.copy_(torch.from_numpy(b))
                conv_idx += 1
            gamma, beta, mean, var = bn.get_weights()
            block.bn.weight.copy_(torch.from_numpy(gamma))
            block.bn.bias.copy_(torch.from_numpy(beta))
            block.bn.running_mean.copy_(torch.from_numpy(mean))
            block.bn.running_var.copy_(torch.from_numpy(var))
        w, b = convs[conv_idx].get_weights()
        model.out.weight.copy_(_keras_to_torch_conv(w))
        model.out.bias.copy_(torch.from_numpy(b))
        for up, keras_up in zip((model.up6, model.up7, model.up8, model.up9), ups):
            w, b = keras_up.get_weights()
            up.conv.weight.copy_(_keras_to_torch_conv(w))
            up.conv.bias.copy_(torch.from_numpy(b))

    model.eval()

    if verify:
        rng = np.random.default_rng(0)
        x = rng.random((1, 256, 256, in_channels), dtype=np.float32)
        keras_out = np.asarray(keras_model.predict(x, verbose=0))
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x).permute(0, 3, 1, 2))
        torch_out = torch_out.permute(0, 2, 3, 1).numpy()
        max_diff = float(np.abs(keras_out - torch_out).max())
        if max_diff > 1e-3:
            raise RuntimeError(
                f"Converted model output differs from Keras (max abs diff {max_diff:.3g})."
            )

    if pt_path is not None:
        save_unet(model, pt_path)
    return model
