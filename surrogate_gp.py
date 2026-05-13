import numpy as np
import cv2
import shutil
import copy
import random
from pathlib import Path
import segmenteverygrain as seg
from synthetic_noise import (
    get_noise_loss,
)
from create_synthetic_images import (
    generate_synthetic_images as generate_synthetic_images_from_script,
    load_image_mask_pairs,
    load_image,
)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from keras.optimizers import Adam
from segmenteverygrain.resnext_model import MaskingResNeXt, weighted_crossentropy_torch

TARGET_PATH = "./real_noisy_images/"
CLEAN_PATH = "./real_clean_images/"
PREDICT_PATH = TARGET_PATH


def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_pairs(pairs, folder):
    folder = reset_dir(folder)
    for img_path, mask_path in pairs:
        shutil.copy2(img_path, folder / Path(img_path).name)
        shutil.copy2(mask_path, folder / Path(mask_path).name)
    return folder


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
    model_weights_file=None,
    use_pretrained=False,
):
    """Train on synthetic multi-res patches and evaluate on held-out real noisy images."""
    workspace = Path(workspace)
    patch_dir = reset_dir(workspace / "patches")

    real_pairs = load_image_mask_pairs(real_noisy_folder)
    if len(real_pairs) < 2:
        raise ValueError("Need at least two real noisy image/mask pairs for validation/test splitting.")
    val_pairs, test_pairs = train_test_split(real_pairs, test_size=0.4, random_state=42)
    real_val_dir = stage_pairs(val_pairs, workspace / "real_val")
    real_test_dir = stage_pairs(test_pairs, workspace / "real_test")

    synthetic_folder = str(synthetic_folder)
    if not synthetic_folder.endswith("/"):
        synthetic_folder = f"{synthetic_folder}/"

    syn_image_dir, syn_mask_dir = seg.patchify_training_data(synthetic_folder, patch_dir / "synthetic")
    val_image_dir, val_mask_dir = seg.patchify_training_data(f"{real_val_dir}/", patch_dir / "real_val")
    test_image_dir, test_mask_dir = seg.patchify_training_data(f"{real_test_dir}/", patch_dir / "real_test")

    syn_images = sorted(glob(syn_image_dir + "/*.png"))
    syn_masks = sorted(glob(syn_mask_dir + "/*.png"))
    val_images = sorted(glob(val_image_dir + "/*.png"))
    val_masks = sorted(glob(val_mask_dir + "/*.png"))
    test_images = sorted(glob(test_image_dir + "/*.png"))
    test_masks = sorted(glob(test_mask_dir + "/*.png"))

    split_dir = patch_dir / "synthetic" / "Patches"
    train_dir = split_dir / "train"

    print("Creating multi-resolution synthetic training data...")
    train_images, train_masks = create_scaled_variants(syn_images, syn_masks, scales, train_dir)

    print(f"Training: {len(train_images)} synthetic patches")
    print(f"Validation: {len(val_images)} real noisy patches")
    print(f"Test: {len(test_images)} real noisy patches")

    if model_family in {"unet", "unet_modified"}:
        train_dataset = build_dataset(train_images, train_masks, augmentation=True)
        val_dataset = build_dataset(val_images, val_masks, augmentation=False)
        test_dataset = build_dataset(test_images, test_masks, augmentation=False)

        if use_pretrained:
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
        "real_val_dir": str(real_val_dir),
        "real_test_dir": str(real_test_dir),
        "model_path": str(model_path),
    }
    return model, metrics

def black_box(
    theta,
    model_family="unet",
    model_weights_file=None,
    use_pretrained=False,
    resolution_weight=0.35,
):
    theta = np.asarray(theta, dtype=float)
    theta_tag = "_".join(f"{value:.4g}" for value in theta)
    workspace = reset_dir("./blackbox_workspace")

    print(f"Running black box for theta={theta_tag} using model_family={model_family}")
    synthetic_folder = generate_synthetic_images_from_script(
        theta,
        input_folder=CLEAN_PATH,
        noise_reference_folder=TARGET_PATH,
        output_folder=workspace / "synthetic_noisy_images",
        seed=42,
    )

    _, metrics = train_model_on_resolutions(
        synthetic_folder,
        real_noisy_folder=TARGET_PATH,
        model_name="synthetic_blackbox",
        workspace=workspace,
        model_family=model_family,
        model_weights_file=model_weights_file,
        use_pretrained=use_pretrained,
    )

    synthetic_pairs = load_image_mask_pairs(synthetic_folder)
    real_val_pairs = load_image_mask_pairs(metrics["real_val_dir"])
    cost = np.zeros((len(synthetic_pairs), len(real_val_pairs)), dtype=np.float32)

    for i, (synthetic_img_path, _) in enumerate(synthetic_pairs):
        synthetic_img = load_image(synthetic_img_path)
        for j, (real_img_path, _) in enumerate(real_val_pairs):
            real_img = load_image(real_img_path)
            cost[i, j] = get_noise_loss(synthetic_img, real_img)

    row_ind, col_ind = linear_sum_assignment(cost)
    resolution_loss = float(np.mean(cost[row_ind, col_ind]))
    objective_value = float(metrics["val_loss"] + resolution_weight * resolution_loss)

    summary = {
        "theta": theta.tolist(),
        "model_family": model_family,
        "model_weights_file": model_weights_file,
        "use_pretrained": use_pretrained,
        "objective": objective_value,
        "resolution_loss": resolution_loss,
        "resolution_weight": resolution_weight,
        "metrics": metrics,
    }
    with open(workspace / "last_blackbox_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        "Black-box metrics: "
        f"objective={objective_value:.6f}, "
        f"val_loss={metrics['val_loss']:.6f}, "
        f"test_loss={metrics['test_loss']:.6f}, "
        f"resolution_loss={resolution_loss:.6f}, "
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
    gp = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=np.ones(n_dim)), n_restarts_optimizer=10)
    gp.fit(X_scaled, y)

    sampler = qmc.LatinHypercube(d=n_dim, seed=42)
    X_test_unit = sampler.random(n=n_test)
    X_test = qmc.scale(X_test_unit, lb, ub)
    X_test_scaled = (X_test - lb) / (ub - lb)

    y_pred, sigma = gp.predict(X_test_scaled, return_std=True)
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
    model_weights_file=None,
    use_pretrained=False,
    resolution_weight=0.35,
):
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
                use_pretrained=use_pretrained,
                resolution_weight=resolution_weight,
            )
        ])
        save_gp_data(data_path, X_train, y_train)

    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} ---")
        print(f"Current dataset size: {len(X_train)}")

        X_scaled = (X_train - lb) / (ub - lb)
        gp, theta_next, pred_mean, pred_std = suggest_next(X_scaled, y_train, n_test=n_test, beta=beta)

        print(f"GP suggests theta: {theta_next}")
        print(f"GP prediction: mean={pred_mean:.4f}, std={pred_std:.4f}")

        print("Evaluating black_box(theta_next)...")
        y_next = black_box(
            theta_next,
            model_family=model_family,
            model_weights_file=model_weights_file,
            use_pretrained=use_pretrained,
            resolution_weight=resolution_weight,
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
    N_ITERATIONS = 5  # Change this to control how many searches to run

    X_final, y_final = run_gp_loop(n_iterations=N_ITERATIONS)

    print("\n=== Final Results ===")
    for i in range(len(X_final)):
        print(f"  [{i}] a={X_final[i,0]:.6f}, b={X_final[i,1]:.6f}, sigma_r={X_final[i,2]:.6f}, l={X_final[i,3]:.4f}, k={X_final[i,4]:.4f} -> f={y_final[i]:.4f}")
    best = np.argmin(y_final)
    print(f"\nBest: {X_final[best]} with f={y_final[best]:.4f}")
