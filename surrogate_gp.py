"""Black-box objective + Bayesian optimization loop for synthetic-data search.

High-level flow:
theta -> synthetic generator -> train chosen model -> eval on real data -> score

The GP loop then uses those scores to suggest the next theta to try.
"""

import numpy as np
import cv2
import shutil
import copy
import random
from pathlib import Path
import segmenteverygrain as seg
from create_synthetic_images import (
    generate_synthetic_images as generate_synthetic_images_from_script,
    load_image_mask_pairs,
)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc

import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from keras.optimizers import Adam
from segmenteverygrain.resnext_model import MaskingResNeXt, weighted_crossentropy_torch

TARGET_PATH = "./real_noisy_images/"
CLEAN_PATH = "./real_clean_images/"
PREDICT_PATH = "./prediction_noisy_images/"


# Workspace helper: recreate a directory from scratch for each run.
def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Split-selected image/mask pairs into a fresh folder on disk.
def stage_pairs(pairs, folder):
    folder = reset_dir(folder)
    for img_path, mask_path in pairs:
        shutil.copy2(img_path, folder / Path(img_path).name)
        shutil.copy2(mask_path, folder / Path(mask_path).name)
    return folder


# Multi-resolution augmentation: downsample then upsample each synthetic patch.
def create_scaled_variants(image_files, mask_files, scales, split_dir):
    """For a given split, creates downsampled+upsampled variants at each scale and saves to disk."""
    split_dir = Path(split_dir)
    all_images = list(image_files)
    all_masks = list(mask_files)

    for scale in scales:
        scale_dir = split_dir / f"res_{scale:.2f}"
        img_dir = scale_dir / "images"
        mask_dir = scale_dir / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in zip(image_files, mask_files):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)

            down_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            up_img = cv2.resize(down_img, (w, h), interpolation=cv2.INTER_CUBIC)

            img_name = Path(img_path).stem
            cv2.imwrite(str(img_dir / f"{img_name}_res{int(scale*100)}.png"), up_img)
            cv2.imwrite(str(mask_dir / f"{img_name}_res{int(scale*100)}.png"), mask)

        all_images += sorted(glob(str(img_dir / "*.png")))
        all_masks += sorted(glob(str(mask_dir / "*.png")))

    return all_images, all_masks


# Torch patch dataset used only for the ResNeXt path.
class PatchDataset(Dataset):
    """Simple paired patch dataset for torch models."""

    def __init__(self, image_files, mask_files, augment=False):
        self.image_files = list(image_files)
        self.mask_files = list(mask_files)
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("L")

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            k = random.randint(0, 3)
            if k:
                img = TF.rotate(img, 90 * k)
                mask = TF.rotate(mask, 90 * k)

        img_t = torch.from_numpy(np.array(img).astype("float32") / 255.0).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(mask).astype("int64"))
        return img_t, mask_t


# TensorFlow dataset builder used by the Keras U-Net paths.
def build_dataset(image_files, mask_files, augmentation=False, batch_size=32, shuffle_buffer=1000):
    """Builds a TF dataset from image and mask file paths."""
    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))

    if augmentation:
        dataset = tf.data.Dataset.from_tensor_slices((
            image_files,
            mask_files,
            tf.Variable([True] * len(image_files), dtype=tf.bool),
        ))

    dataset = dataset.map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Shared evaluation helper for torch models.
def evaluate_torch_model(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = weighted_crossentropy_torch(preds, masks, device=device)
            running_loss += loss.item() * imgs.size(0)

            pred_labels = torch.argmax(preds, dim=1)
            correct += (pred_labels == masks).sum().item()
            total += masks.numel()

    return {
        "loss": float(running_loss / max(1, len(dataloader.dataset))),
        "accuracy": float(correct / max(1, total)),
    }


def train_model_on_resolutions(
    synthetic_folder,
    real_noisy_folder=TARGET_PATH,
    model_name="synthetic_blackbox",
    scales=(0.5, 0.75, 1.0),
    workspace="./blackbox_workspace",
    model_family="unet",
    model_weights_file="./models/seg_model.keras",
    use_pretrained=True,
):
    """Train on synthetic multi-res patches and evaluate on held-out real noisy images."""
    workspace = Path(workspace)
    patch_dir = reset_dir(workspace / "patches")

    syn_image_dir, syn_mask_dir = seg.patchify_training_data(synthetic_folder, Path(patch_dir) / "synthetic")
    real_image_dir, real_mask_dir = seg.patchify_training_data(real_noisy_folder, Path(patch_dir) / "real")

    syn_images = sorted(glob(syn_image_dir + "/*.png"))
    syn_masks = sorted(glob(syn_mask_dir + "/*.png"))
    real_images = sorted(glob(real_image_dir + "/*.png"))
    real_masks = sorted(glob(real_mask_dir + "/*.png"))

    train_val_syn_images, test_syn_images, train_val_syn_masks, test_syn_masks = train_test_split(
        syn_images, syn_masks, test_size=0.15, random_state=42
    )
    train_syn_images, val_syn_images, train_syn_masks, val_syn_masks = train_test_split(
        train_val_syn_images, train_val_syn_masks, test_size=0.25, random_state=42
    )

    train_val_real_images, test_real_images, train_val_real_masks, test_real_masks = train_test_split(
        real_images, real_masks, test_size=0.15, random_state=42
    )
    train_real_images, val_real_images, train_real_masks, val_real_masks = train_test_split(
        train_val_real_images, train_val_real_masks, test_size=0.25, random_state=42
    )

    split_dir = Path(patch_dir) / "synthetic" / "Patches"
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    test_dir = split_dir / "test"

    print("Creating multi-resolution synthetic training data...")
    train_syn_images, train_syn_masks = create_scaled_variants(train_syn_images, train_syn_masks, scales, train_dir)
    print("Creating multi-resolution synthetic validation data...")
    val_syn_images, val_syn_masks = create_scaled_variants(val_syn_images, val_syn_masks, scales, val_dir)
    print("Creating multi-resolution synthetic test data...")
    test_syn_images, test_syn_masks = create_scaled_variants(test_syn_images, test_syn_masks, scales, test_dir)

    train_images = train_syn_images + train_real_images
    train_masks = train_syn_masks + train_real_masks
    val_images = val_syn_images + val_real_images
    val_masks = val_syn_masks + val_real_masks
    test_images = test_syn_images + test_real_images
    test_masks = test_syn_masks + test_real_masks

    print(f"Training: {len(train_images)} images ({len(train_syn_images)} synthetic + {len(train_real_images)} real)")
    print(f"Validation: {len(val_images)} images ({len(val_syn_images)} synthetic + {len(val_real_images)} real)")
    print(f"Test: {len(test_images)} images ({len(test_syn_images)} synthetic + {len(test_real_images)} real)")

    if model_family in {"unet", "unet_modified"}:
        # Keras path: either start from the repo constructors or fine-tune a saved model.
        print("Using Unet")
        train_dataset = build_dataset(train_images, train_masks, augmentation=True)
        val_dataset = build_dataset(val_images, val_masks, augmentation=False)
        test_dataset = build_dataset(test_images, test_masks, augmentation=False)

        if use_pretrained:
            print("Using pretrained")
            if model_weights_file is None:
                raise ValueError("model_weights_file is required when use_pretrained=True.")
            model = seg.create_and_train_model_from_pretrained(
                model_weights_file,
                train_dataset,
                val_dataset,
                test_dataset,
                epochs=80,
                learning_rate=1e-2,
                model_type=model_family,
                save_plot_path=f"loss_plots/training_loss_plot_{model_name}.png",
                show_plot=False,
                use_reduce_lr=True,
            )
        else:
            if model_family == "unet_modified":
                model = seg.UnetModified()
            else:
                model = seg.Unet()
            model.compile(
                optimizer=Adam(learning_rate=1e-2),
                loss=seg.weighted_crossentropy,
                metrics=["accuracy"],
            )
            model.fit(train_dataset, epochs=80, validation_data=val_dataset)
            model.evaluate(test_dataset, verbose=0)

        model_path = Path("models") / f"{model_name}.keras"
        model.save(model_path)

        val_metrics = model.evaluate(val_dataset, verbose=0, return_dict=True)
        test_metrics = model.evaluate(test_dataset, verbose=0, return_dict=True)
    elif model_family == "resnext":
        # Torch path: mirrors the notebook training template in callable form.
        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        train_loader = DataLoader(PatchDataset(train_images, train_masks, augment=True), batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(PatchDataset(val_images, val_masks, augment=False), batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(PatchDataset(test_images, test_masks, augment=False), batch_size=8, shuffle=False, num_workers=0)

        model = MaskingResNeXt(num_classes=3, pretrained=use_pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        best_model_state = None
        best_val_loss = float("inf")

        for epoch in range(20):
            model.train()
            running_loss = 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = weighted_crossentropy_torch(preds, masks, device=device)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            val_metrics = evaluate_torch_model(model, val_loader, device)
            scheduler.step(val_metrics["loss"])
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_state = copy.deepcopy(model.state_dict())

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model_path = Path("models") / f"{model_name}.pth"
        torch.save(model.state_dict(), model_path)

        val_metrics = evaluate_torch_model(model, val_loader, device)
        test_metrics = evaluate_torch_model(model, test_loader, device)
    else:
        raise ValueError(
            f"Unsupported model_family '{model_family}'. "
            "Choose from 'unet', 'unet_modified', or 'resnext'."
        )
    metrics = {
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "model_path": str(model_path),
    }
    return model, metrics

def dice_loss(predicted_probs, true_masks, eps=1e-7):
    """Multi-class Dice loss averaged over images. Lower = more similar."""
    total = 0.0
    for pred, true in zip(predicted_probs, true_masks):
        h, w = min(pred.shape[0], true.shape[0]), min(pred.shape[1], true.shape[1])
        pred, true = pred[:h, :w], true[:h, :w]
        one_hot = np.eye(pred.shape[-1])[true]
        intersection = np.sum(pred * one_hot, axis=(0, 1))
        union = np.sum(pred + one_hot, axis=(0, 1))
        total += float(1 - np.mean((2 * intersection + eps) / (union + eps)))
    return total / max(1, len(predicted_probs))


def count_penalty(predicted_probs, true_masks):
    """Normalized absolute difference in number of grain instances. Lower = more similar."""
    total = 0.0
    for pred, true in zip(predicted_probs, true_masks):
        h, w = min(pred.shape[0], true.shape[0]), min(pred.shape[1], true.shape[1])
        pred_label = np.argmax(pred[:h, :w], axis=-1).astype(np.uint8)
        true_label = true[:h, :w].astype(np.uint8)
        n_pred = max(cv2.connectedComponents((pred_label == 1).astype(np.uint8))[0] - 1, 0)
        n_true = max(cv2.connectedComponents((true_label == 1).astype(np.uint8))[0] - 1, 0)
        total += abs(n_pred - n_true) / max(n_true, 1)
    return total / max(1, len(predicted_probs))


def compute_mask_loss(predicted_probs, true_masks, dice_weight=1.0, count_weight=3.1664):
    """Composite loss: Dice (per-pixel overlap) + count penalty (grain consistency)."""
    dice = dice_loss(predicted_probs, true_masks)
    count = count_penalty(predicted_probs, true_masks)
    return dice_weight * dice + count_weight * count


def evaluate_model_masks(model, image_dir, patch_dir, model_family, device=None, tile_size=256):
    """Run model inference on images and return per-image predictions + true masks."""
    pairs = load_image_mask_pairs(image_dir)
    pair_dir = stage_pairs(pairs, Path(patch_dir) / "staged")
    staged_pairs = load_image_mask_pairs(str(pair_dir))

    if model_family in {"unet", "unet_modified"}:
        pred_probs_list = []
        true_masks_list = []
        for img_path, mask_path in staged_pairs:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = seg.predict_image_mirror(img, model, tile_size)
            pred_probs_list.append(pred)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_masks_list.append(mask)
        return pred_probs_list, true_masks_list

    elif model_family == "resnext":
        img_dir, mask_dir = seg.patchify_training_data(f"{pair_dir}/", Path(patch_dir) / "patched")
        image_files = sorted(glob(img_dir + "/*.png"))
        mask_files = sorted(glob(mask_dir + "/*.png"))

        if device is None:
            device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
        loader = DataLoader(
            PatchDataset(image_files, mask_files, augment=False),
            batch_size=8, shuffle=False, num_workers=0,
        )
        model.eval()
        all_preds = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(device)
                preds = torch.softmax(model(imgs), dim=1).cpu().numpy()
                all_preds.append(preds)
        pred_probs = np.concatenate(all_preds, axis=0)
        pred_probs = np.transpose(pred_probs, (0, 2, 3, 1))

        true_masks = []
        for mp in mask_files:
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            true_masks.append(mask)
        return [pred_probs[i] for i in range(len(pred_probs))], true_masks
    else:
        raise ValueError(f"Unsupported model_family '{model_family}'")


def black_box(
    theta,
    model_family="unet",
    model_weights_file=None,
    use_pretrained=False,
    dice_weight=1.0,
    count_weight=3.1664,
):
    """Evaluate one candidate theta and return a single scalar score."""
    theta = np.asarray(theta, dtype=float)
    theta_tag = "_".join(f"{value:.4g}" for value in theta)
    workspace = reset_dir("./blackbox_workspace")

    # 1) Generate synthetic noisy image/mask pairs from the candidate theta.
    print(f"Running black box for theta={theta_tag} using model_family={model_family}")
    synthetic_folder = generate_synthetic_images_from_script(
        theta,
        input_folder=CLEAN_PATH,
        noise_reference_folder=TARGET_PATH,
        output_folder=workspace / "synthetic_noisy_images",
        seed=42,
    )

    # 2) Train the chosen model family on synthetic patches and evaluate on real data.
    model, metrics = train_model_on_resolutions(
        synthetic_folder = synthetic_folder,
        real_noisy_folder=TARGET_PATH,
        model_name="clean_blackbox",
        workspace=workspace,
        model_family=model_family,
        model_weights_file=model_weights_file,
        use_pretrained=use_pretrained,
    )

    # 3) Predict masks on PREDICT_PATH images and compare with ground truth masks.
    pred_probs, true_masks = evaluate_model_masks(
        model,
        image_dir=PREDICT_PATH,
        patch_dir=workspace / "eval_patches",
        model_family=model_family,
    )
    mask_loss = compute_mask_loss(pred_probs, true_masks, dice_weight, count_weight)

    # 4) Final score: composite mask prediction loss.
    objective_value = float(mask_loss)

    summary = {
        "theta": theta.tolist(),
        "model_family": model_family,
        "model_weights_file": model_weights_file,
        "use_pretrained": use_pretrained,
        "objective": objective_value,
        "dice_loss": dice_loss(pred_probs, true_masks),
        "count_penalty": count_penalty(pred_probs, true_masks),
        "dice_weight": dice_weight,
        "count_weight": count_weight,
        "metrics": metrics,
    }
    with open(workspace / "last_blackbox_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "Black-box metrics: "
        f"objective={objective_value:.6f}, "
        f"val_loss={metrics['val_loss']:.6f}, "
        f"test_loss={metrics['test_loss']:.6f}, "
        f"dice={dice_loss(pred_probs, true_masks):.6f}, "
        f"count_pen={count_penalty(pred_probs, true_masks):.6f}, "
        f"test_accuracy={metrics['test_accuracy']:.6f}"
    )
    return objective_value


# Parameter bounds for theta = [a, b, sigma_r, l, k] from synthetic_noise.py differential_evolution bounds
bounds = np.array([
    [1e-6, 0.2],    # a
    [1e-6, 0.05],   # b
    [1e-6, 0.05],   # sigma_r
    [0.3, 8.0],     # l
    [1e-6, 0.1],    # k
])
n_dim = bounds.shape[0]
lb, ub = bounds[:, 0], bounds[:, 1]

DATA_PATH = "gp_data.json"

def load_gp_data(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        X = np.array(data["X"])
        y = np.array(data["y"])
        print(f"Loaded {X.shape[0]} previous data points from {path}")
        return X, y
    return None, None

def save_gp_data(path, X, y):
    with open(path, "w") as f:
        json.dump({"X": X.tolist(), "y": y.tolist()}, f, indent=2)

def suggest_next(X_scaled, y, n_test=500, beta=1.96):
    """Fit a GP surrogate and propose the next theta to evaluate."""
    gp = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=np.ones(n_dim)), n_restarts_optimizer=10)
    gp.fit(X_scaled, y)

    sampler = qmc.LatinHypercube(d=n_dim, seed=42)
    X_test_unit = sampler.random(n=n_test)
    X_test = qmc.scale(X_test_unit, lb, ub)
    X_test_scaled = (X_test - lb) / (ub - lb)

    y_pred, sigma = gp.predict(X_test_scaled, return_std=True)
    # Lower objective is better, so use a lower-confidence-bound style acquisition.
    acquisition = y_pred - beta * sigma
    best_idx = np.argmin(acquisition)
    return gp, X_test[best_idx], y_pred[best_idx], sigma[best_idx]

def run_gp_loop(
    n_iterations,
    initial_theta=None,
    data_path=DATA_PATH,
    n_test=500,
    beta=1.96,
    model_family="unet",
    model_weights_file="./models/seg_model.keras",
    use_pretrained=True
):
    """Template Bayesian optimization loop around the expensive black box."""
    X_prev, y_prev = load_gp_data(data_path)

    if X_prev is not None and len(X_prev) > 0:
        X_train = X_prev.copy()
        y_train = y_prev.copy()
        print(f"Continuing with {len(X_train)} existing data points")
    else:
        if initial_theta is None:
            theta_initial = np.array([0.0023589515117326183, 0.001712502743955444, 0.0006997093027690107, 0.7603779994083678, 0.07404317063233228])
        else:
            theta_initial = np.array(initial_theta)
        print(f"Evaluating initial theta: {theta_initial}")
        X_train = theta_initial.reshape(1, -1)
        y_train = np.array([
            black_box(
                theta_initial,
                model_family=model_family,
                model_weights_file=model_weights_file,
                use_pretrained=use_pretrained
            )
        ])
        save_gp_data(data_path, X_train, y_train)

    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} ---")
        print(f"Current dataset size: {len(X_train)}")

        # Fit/update the surrogate, ask it for the next theta, then evaluate.
        X_scaled = (X_train - lb) / (ub - lb)
        gp, theta_next, pred_mean, pred_std = suggest_next(X_scaled, y_train, n_test=n_test, beta=beta)

        print(f"GP suggests theta: {theta_next}")
        print(f"GP prediction: mean={pred_mean:.4f}, std={pred_std:.4f}")

        print("Evaluating black_box(theta_next)...")
        y_next = black_box(
            theta_next,
            model_family=model_family,
            model_weights_file=model_weights_file,
            use_pretrained=use_pretrained
        )
        print(f"Result: f(theta) = {y_next}")

        X_train = np.vstack([X_train, theta_next])
        y_train = np.append(y_train, y_next)
        save_gp_data(data_path, X_train, y_train)

        best_idx = np.argmin(y_train)
        print(f"Best theta so far (iter {best_idx}): f={y_train[best_idx]:.4f}")
        print(f"Data saved to {data_path}")

    return X_train, y_train

if __name__ == "__main__":
    N_ITERATIONS = 1  # Change this to control how many searches to run

    X_final, y_final = run_gp_loop(n_iterations=N_ITERATIONS)

    print("\n=== Final Results ===")
    for i in range(len(X_final)):
        print(f"  [{i}] a={X_final[i,0]:.6f}, b={X_final[i,1]:.6f}, sigma_r={X_final[i,2]:.6f}, l={X_final[i,3]:.4f}, k={X_final[i,4]:.4f} -> f={y_final[i]:.4f}")
    best = np.argmin(y_final)
    print(f"\nBest: {X_final[best]} with f={y_final[best]:.4f}")
