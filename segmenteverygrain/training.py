"""
Training utilities for the segmenteverygrain U-Net (PyTorch).

This module covers the full training workflow: cutting large annotated images
into patches (:func:`patchify_training_data`), loading and augmenting patches
(:class:`GrainPatchDataset`, :func:`augment_pair`), building data loaders
(:func:`create_train_val_test_data`), the weighted cross-entropy loss
(:func:`weighted_crossentropy`), the training loop (:func:`train_model`,
:func:`create_and_train_model`), and evaluation (:func:`evaluate_model`).

Conventions
-----------
Images are float32 tensors of shape ``(N, 3, H, W)`` with values in ``[0, 1]``.
Masks are int64 tensors of shape ``(N, H, W)`` holding class indices
(0 = background, 1 = grain, 2 = grain boundary). The model returns logits of
shape ``(N, 3, H, W)``.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from glob import glob
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TVF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .unet import DEFAULT_CLASS_WEIGHTS, UNet, get_device, load_unet

PATCH_SIZE = 256
CLASS_NAMES = ["background", "grain", "boundary"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def weighted_crossentropy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    label_smoothing: float = 0.1,
    class_weights: Sequence[float] = DEFAULT_CLASS_WEIGHTS,
) -> torch.Tensor:
    """
    Weighted cross-entropy loss with label smoothing, computed from logits.

    Each pixel's cross-entropy is multiplied by the weight of its true class;
    the default weights (0.6, 1.0, 5.0) emphasize grains and especially grain
    boundaries. Label smoothing is applied to the targets to prevent
    overconfident predictions.

    Parameters
    ----------
    y_true : tensor
        True labels, either class indices of shape ``(N, H, W)`` or one-hot
        encoded of shape ``(N, C, H, W)``.
    y_pred : tensor
        Predicted logits (pre-softmax) of shape ``(N, C, H, W)``.
    label_smoothing : float
        Label smoothing factor. 0.0 means no smoothing.
    class_weights : sequence of float
        One weight per class.

    Returns
    -------
    loss : tensor
        Scalar loss (mean over all pixels).
    """
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    y_true = torch.as_tensor(y_true, device=y_pred.device)
    num_classes = y_pred.shape[1]
    if y_true.ndim == y_pred.ndim - 1:
        one_hot = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float()
    else:
        one_hot = y_true.float()
    smoothed = one_hot * (1.0 - label_smoothing) + label_smoothing / num_classes
    log_probs = F.log_softmax(y_pred, dim=1)
    unweighted = -(smoothed * log_probs).sum(dim=1)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=y_pred.device)
    pixel_weights = (weights.view(1, -1, 1, 1) * one_hot).sum(dim=1)
    return (pixel_weights * unweighted).mean()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _nearest_resize_indices(out_size: int, in_size: int) -> np.ndarray:
    """Source indices for a half-pixel-centered nearest-neighbor resize."""
    idx = np.floor((np.arange(out_size) + 0.5) * in_size / out_size).astype(int)
    return np.clip(idx, 0, in_size - 1)


def augment_pair(
    image: np.ndarray, mask: np.ndarray, rng: Optional[np.random.Generator] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly augment an image/mask pair.

    Geometric augmentations (random flips, 90-degree rotations, and a random
    crop-and-zoom with probability 0.5) are applied identically to the image
    and the mask. Color augmentations (brightness, contrast, saturation, hue,
    Gaussian noise, each with probability 0.5, and Gaussian blur with
    probability 0.3) are applied to the image only.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(H, W, 3)``, float32, values in ``[0, 1]``.
    mask : numpy.ndarray
        Mask of shape ``(H, W)`` with integer class indices.
    rng : numpy.random.Generator, optional
        Random number generator (a fresh one is used when None).

    Returns
    -------
    image : numpy.ndarray
        Augmented image, float32 ``(H, W, 3)`` in ``[0, 1]``.
    mask : numpy.ndarray
        Augmented mask, same dtype as the input.
    """
    if rng is None:
        rng = np.random.default_rng()
    height, width = mask.shape

    # --- Geometric augmentations (image and mask together) ---
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1, :], mask[::-1, :]
    k = int(rng.integers(0, 4))
    image, mask = np.rot90(image, k, axes=(0, 1)), np.rot90(mask, k, axes=(0, 1))
    if rng.random() < 0.5:
        scale = rng.uniform(0.7, 0.95)
        crop_h = int(round(height * scale))
        crop_w = int(round(width * scale))
        off_h = int(rng.integers(0, max(height - crop_h, 1)))
        off_w = int(rng.integers(0, max(width - crop_w, 1)))
        rows = off_h + _nearest_resize_indices(height, crop_h)
        cols = off_w + _nearest_resize_indices(width, crop_w)
        image = image[rows][:, cols]
        mask = mask[rows][:, cols]
    image = np.ascontiguousarray(image, dtype=np.float32)
    mask = np.ascontiguousarray(mask)

    # --- Color augmentations (image only), as a (3, H, W) tensor ---
    img = torch.from_numpy(image).permute(2, 0, 1)
    if rng.random() < 0.5:
        img = img + float(rng.uniform(-0.2, 0.2))
    if rng.random() < 0.5:
        factor = float(rng.uniform(0.8, 1.2))
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * factor + mean
    if rng.random() < 0.5:
        img = TVF.adjust_saturation(img.clamp(0.0, 1.0), float(rng.uniform(0.7, 1.3)))
    if rng.random() < 0.5:
        img = TVF.adjust_hue(img.clamp(0.0, 1.0), float(rng.uniform(-0.05, 0.05)))
    if rng.random() < 0.5:
        noise = torch.from_numpy(rng.normal(0.0, 0.02, size=img.shape).astype(np.float32))
        img = img + noise
    if rng.random() < 0.3:
        img = TVF.gaussian_blur(img, kernel_size=5, sigma=float(rng.uniform(0.5, 1.5)))
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).contiguous().numpy(), mask


class GrainPatchDataset(Dataset):
    """
    Dataset of image patches and their label masks stored as PNG files.

    Parameters
    ----------
    image_files : sequence of str
        Paths to the image patches (RGB PNG).
    mask_files : sequence of str
        Paths to the corresponding masks (8-bit PNG with values 0, 1, 2).
    augment : bool
        Whether to apply :func:`augment_pair` to each sample.

    Notes
    -----
    ``__getitem__`` returns ``(image, mask)`` with ``image`` a float32 tensor
    of shape ``(3, H, W)`` in ``[0, 1]`` and ``mask`` an int64 tensor of shape
    ``(H, W)``.
    """

    def __init__(
        self, image_files: Sequence[str], mask_files: Sequence[str], augment: bool = False
    ) -> None:
        if len(image_files) != len(mask_files):
            raise ValueError(
                f"Got {len(image_files)} image files but {len(mask_files)} mask files."
            )
        self.image_files = list(image_files)
        self.mask_files = list(mask_files)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.image_files[index]) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(self.mask_files[index]) as im:
            mask = np.asarray(im.convert("L")).astype(np.int64)
        if self.augment:
            image, mask = augment_pair(image, mask)
        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous()
        return image_t, torch.from_numpy(np.ascontiguousarray(mask))


def patchify_training_data(
    input_dir: str, patch_dir: str, patch_size: int = PATCH_SIZE, stride: int = PATCH_SIZE // 2
) -> Optional[tuple[str, str]]:
    """
    Extracts patches from training images and labels, and saves them to the specified directory.

    Parameters
    ----------
    input_dir : str
        The directory containing the input images and labels.
    patch_dir : str
        The directory where the patches will be saved.
    patch_size : int, optional
        Size of the square patches in pixels. Defaults to 256.
    stride : int, optional
        Step between patches in pixels. Defaults to 128 (50% overlap).

    Returns
    -------
    tuple
        A tuple containing the paths to the directories where the image patches and label patches are saved.

    Notes
    -----
    - The function expects the input directory to contain files with 'image' and 'mask' in their filenames.
    - The function creates subdirectories 'images' and 'labels' within the specified patch directory to save the patches.
    - If the input directory does not exist or contains no matching files, the function will print a warning and return.
    """
    if not os.path.exists(input_dir):
        print(f"Warning: The directory {input_dir} does not exist.")
        return None

    images = sorted(glob(os.path.join(input_dir, "*image*")))
    labels = sorted(glob(os.path.join(input_dir, "*mask*")))

    if not images or not labels:
        print(f"Warning: No files containing 'image' or 'mask' found in {input_dir}.")
        return None

    patches_dir = os.path.join(patch_dir, "Patches")
    images_dir = os.path.join(patches_dir, "images")
    labels_dir = os.path.join(patches_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    def write_patches(files, out_dir, channels):
        start_no = 0
        for fname in tqdm(files):
            with Image.open(fname) as im:
                large_image = np.asarray(im.convert("RGB"))
            if channels == 1:
                large_image = large_image[:, :, 0]  # only keep one layer
            height, width = large_image.shape[:2]
            count = 0
            for row in range(0, height - patch_size + 1, stride):
                for col in range(0, width - patch_size + 1, stride):
                    patch = large_image[row : row + patch_size, col : col + patch_size]
                    imname = os.path.join(out_dir, "im%03d.png" % (start_no + count))
                    Image.fromarray(patch.astype(np.uint8)).save(imname)
                    count += 1
            start_no += count

    write_patches(images, images_dir, channels=3)
    write_patches(labels, labels_dir, channels=1)
    return images_dir, labels_dir


def create_train_val_test_data(
    image_dir: str,
    mask_dir: str,
    augmentation: bool = True,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits image and mask data into training, validation, and test datasets, with optional augmentation.

    The patches are split 60% / 25% / 15% (training / validation / test) with a
    fixed random seed, so repeated calls give the same split.

    Parameters
    ----------
    image_dir : str
        Directory containing the image files.
    mask_dir : str
        Directory containing the mask files.
    augmentation : bool, optional
        If True, applies data augmentation to the training dataset (default is True).
    batch_size : int, optional
        Batch size of the data loaders. Defaults to 32.
    num_workers : int, optional
        Number of background processes used to load and augment patches.
        When None (default), up to 4 workers are used on platforms where
        processes are forked (Linux) and 0 (loading in the main process) on
        macOS and Windows, where spawned worker processes import ``torch``
        before SciPy and can abort with a duplicate-OpenMP-runtime error in
        conda environments. Set it explicitly to override.

    Returns
    -------
    train_dataset : torch.utils.data.DataLoader
        Data loader for training (shuffled).
    val_dataset : torch.utils.data.DataLoader
        Data loader for validation.
    test_dataset : torch.utils.data.DataLoader
        Data loader for testing.
    """
    image_files = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))
    if not image_files:
        raise FileNotFoundError(f"No PNG patches found in {image_dir}")
    if num_workers is None:
        forks = multiprocessing.get_start_method(allow_none=True) == "fork" or (
            multiprocessing.get_start_method(allow_none=True) is None
            and "fork" in multiprocessing.get_all_start_methods()
            and sys.platform.startswith("linux")
        )
        num_workers = min(4, os.cpu_count() or 1) if forks else 0

    # First, split the data into training + validation and test sets
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=0.15, random_state=42
    )
    # Then split the training + validation set into training and validation sets
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=0.25, random_state=42
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        GrainPatchDataset(train_images, train_masks, augment=augmentation),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(GrainPatchDataset(val_images, val_masks), **loader_kwargs)
    test_loader = DataLoader(GrainPatchDataset(test_images, test_masks), **loader_kwargs)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _run_epoch(model, loader, device, optimizer=None, desc=""):
    """One pass over ``loader``; trains when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    with torch.set_grad_enabled(training):
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = weighted_crossentropy(masks, logits)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n_pixels = masks.numel()
            total_loss += loss.item() * n_pixels
            total_correct += (logits.argmax(dim=1) == masks).sum().item()
            total_pixels += n_pixels
    return total_loss / max(total_pixels, 1), total_correct / max(total_pixels, 1)


def train_model(
    model: UNet,
    train_dataset: DataLoader,
    val_dataset: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    Train a U-Net with the Adam optimizer and :func:`weighted_crossentropy`.

    Parameters
    ----------
    model : UNet
        The model to train (modified in place).
    train_dataset : DataLoader
        Training data loader (see :func:`create_train_val_test_data`).
    val_dataset : DataLoader, optional
        Validation data loader, evaluated after every epoch.
    epochs : int
        Number of epochs. Defaults to 100.
    learning_rate : float
        Adam learning rate. Defaults to 1e-3.
    device : str or torch.device, optional
        Device to train on; auto-detected (CUDA, MPS, CPU) when None.

    Returns
    -------
    history : dict
        Per-epoch lists under the keys ``"loss"``, ``"accuracy"`` and, when a
        validation loader is given, ``"val_loss"`` and ``"val_accuracy"``.
    """
    device = get_device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: dict[str, list[float]] = {"loss": [], "accuracy": []}
    if val_dataset is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []
    for epoch in range(1, epochs + 1):
        loss, acc = _run_epoch(
            model, train_dataset, device, optimizer, desc=f"epoch {epoch}/{epochs}"
        )
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        message = f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}"
        if val_dataset is not None:
            val_loss, val_acc = _run_epoch(model, val_dataset, device, desc="validation")
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            message += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
        print(message)
    model.eval()
    return history


def plot_training_history(history: dict):
    """
    Plot training (and validation) loss and accuracy per epoch.

    Parameters
    ----------
    history : dict
        As returned by :func:`train_model`.

    Returns
    -------
    fig, axes
        The matplotlib figure and its two axes.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(history["loss"], label="training")
    axes[1].plot(history["accuracy"], label="training")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="validation")
        axes[1].plot(history["val_accuracy"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[0].legend()
    axes[1].legend()
    return fig, axes


def create_and_train_model(
    train_dataset: DataLoader,
    val_dataset: DataLoader,
    test_dataset: DataLoader,
    model_file: Optional[str] = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[Union[str, torch.device]] = None,
) -> UNet:
    """
    Create (or load) a U-Net model, train it, and evaluate it on the test set.

    Parameters
    ----------
    train_dataset : DataLoader
        Training data loader.
    val_dataset : DataLoader
        Validation data loader.
    test_dataset : DataLoader
        Test data loader.
    model_file : str, optional
        Path to a saved model (``.pt``) whose weights are used as the starting
        point, e.g. for fine-tuning. A Keras ``.keras`` file is converted on
        the fly (requires Keras and TensorFlow). When None, a new model is
        trained from scratch.
    epochs : int, optional
        Number of epochs to train the model (default is 100).
    learning_rate : float, optional
        Adam learning rate (default is 1e-3).
    device : str or torch.device, optional
        Device to train on; auto-detected when None.

    Returns
    -------
    model : UNet
        Trained U-Net model. Save it with :func:`segmenteverygrain.save_unet`.

    Notes
    -----
    The function plots the training and validation loss and accuracy over
    epochs and prints the test-set metrics (see :func:`evaluate_model`).
    """
    device = get_device(device)
    if model_file:
        if model_file.endswith(".keras"):
            from .unet import convert_keras_model

            print(f"converting Keras model {model_file} to PyTorch...")
            model = convert_keras_model(model_file)
        else:
            model = load_unet(model_file, device=device)
    else:
        model = UNet()
    history = train_model(
        model, train_dataset, val_dataset, epochs=epochs,
        learning_rate=learning_rate, device=device,
    )
    plot_training_history(history)
    evaluate_model(model, test_dataset, device=device)
    return model


@torch.no_grad()
def evaluate_model(
    model: UNet,
    dataset: DataLoader,
    class_names: Optional[list[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """
    Evaluate a trained U-Net model on a dataset, reporting overall accuracy,
    loss, and per-class IoU scores.

    Parameters
    ----------
    model : UNet
        The trained model.
    dataset : DataLoader
        The dataset to evaluate on (e.g. the test dataset).
    class_names : list of str, optional
        Names for each output class. Defaults to
        ``["background", "grain", "boundary"]``.
    device : str or torch.device, optional
        Device to run on; defaults to the model's device.

    Returns
    -------
    results : dict
        Dictionary with keys ``"loss"``, ``"accuracy"``, ``"mean_iou"``,
        and ``"per_class_iou"`` (a dict mapping class name to IoU).
    """
    if class_names is None:
        class_names = list(CLASS_NAMES)
    num_classes = len(class_names)
    device = next(model.parameters()).device if device is None else get_device(device)
    model.to(device)
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    n_batches = 0
    for images, masks in tqdm(dataset, desc="evaluating model"):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total_loss += weighted_crossentropy(masks, logits).item()
        n_batches += 1
        pred_classes = logits.argmax(dim=1).cpu().numpy().ravel()
        true_classes = masks.cpu().numpy().ravel()
        confusion += np.bincount(
            true_classes * num_classes + pred_classes, minlength=num_classes**2
        ).reshape(num_classes, num_classes)

    per_class_iou = {}
    for c in range(num_classes):
        tp = confusion[c, c]
        fn = confusion[c, :].sum() - tp
        fp = confusion[:, c].sum() - tp
        denom = tp + fn + fp
        per_class_iou[class_names[c]] = float(tp / denom) if denom > 0 else 0.0

    total_pixels = confusion.sum()
    accuracy = float(np.trace(confusion) / total_pixels) if total_pixels > 0 else 0.0
    mean_iou = float(np.mean(list(per_class_iou.values())))
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

    print(f"\nTest loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    for name, iou in per_class_iou.items():
        print(f"  {name} IoU: {iou:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
    }
