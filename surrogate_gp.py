import numpy as np
import cv2
from pathlib import Path
import segmenteverygrain as seg
from synthetic_noise import (
    NoiseParams,
    make_noisy_training_pair,
    load_images_from_folder,
    percentile_normalize,
)
from create_synthetic_images import load_image_mask_pairs, load_image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import qmc
import json

TARGET_PATH = "./real_noisy_images/"
CLEAN_PATH = "./real_clean_images/"
PREDICT_PATH = TARGET_PATH

def generate_synthetic_images(theta, clean_folder=CLEAN_PATH, noise_ref_folder=TARGET_PATH, output_folder="./synthetic_noisy_images/"):
    """Takes in theta and generates synthetic noisy images from clean image-mask pairs."""
    params = NoiseParams(
        a=float(theta[0]),
        b=float(theta[1]),
        sigma_r=float(theta[2]),
        l=float(theta[3]),
        k=float(theta[4]),
    )

    pairs = load_image_mask_pairs(clean_folder)

    _, noise_paths = load_images_from_folder(noise_ref_folder)
    sample_noisy = cv2.imread(noise_paths[0], cv2.IMREAD_GRAYSCALE)
    target_shape = sample_noisy.shape

    rng = np.random.default_rng(42)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in pairs:
        img_name = Path(img_path).stem
        clean_img = load_image(img_path)
        clean_mask = load_image(mask_path)

        noisy_img, mask_ds, clean_ds = make_noisy_training_pair(
            clean_img=clean_img,
            clean_mask=clean_mask,
            target_shape=target_shape,
            params=params,
            rng=rng,
        )

        noisy_path = Path(output_folder) / f"noisy{img_name[:-5]}image.png"
        mask_path_out = Path(output_folder) / f"noisy{img_name[:-5]}mask.png"

        cv2.imwrite(str(noisy_path), (noisy_img * 255).astype(np.uint8))
        cv2.imwrite(str(mask_path_out), (mask_ds * 255).astype(np.uint8))

    return output_folder


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


def train_model_on_resolutions(synthetic_folder, real_noisy_folder="./testnoisyimages/", model_name="synthetic_blackbox", scales=[0.5, 0.75, 1.0]):
    """Trains a model on multi-res synthetic images plus original-res real noisy images."""
    patch_dir = "patches/"
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

    train_dataset = build_dataset(train_images, train_masks, augmentation=True)
    val_dataset = build_dataset(val_images, val_masks, augmentation=False)
    test_dataset = build_dataset(test_images, test_masks, augmentation=False)

    model = seg.create_and_train_model_from_pretrained(
        "models/seg_model.keras",
        train_dataset,
        val_dataset,
        test_dataset,
        epochs=80,
        learning_rate=1e-2,
        model_type="unet",
        save_plot_path=f"loss_plots/training_loss_plot_{model_name}.png",
        show_plot=False,
        use_reduce_lr=True,
    )

    model.save(f"models/{model_name}.keras")
    return model

def black_box(theta):
    # Takes in theta and generates images, using create_synthetic_images.py
    synthetic_folder = generate_synthetic_images(theta)

    # Trains a model on those synthetic images at several resolutions plus the real noisy images, from train_variant.py
    model = train_model_on_resolutions(synthetic_folder,real_noisy_folder=TARGET_PATH)

    # Runs predictions on just the real noisy images, PREDICT_PATH is the dir of image/mask pairs that the model will predict on
    #predictions = placeholder1(model,PREDICT_PATH)

    # Evaluates the "closeness" to those predictions
    #return placeholder2output(predictions,PREDICT_PATH)


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
    acquisition = y_pred + beta * sigma
    best_idx = np.argmax(acquisition)
    return gp, X_test[best_idx], y_pred[best_idx], sigma[best_idx]

def run_gp_loop(n_iterations, initial_theta=None, data_path=DATA_PATH, n_test=500, beta=1.96):
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
        y_train = np.array([black_box(theta_initial)])
        save_gp_data(data_path, X_train, y_train)

    for i in range(n_iterations):
        print(f"\n--- Iteration {i+1}/{n_iterations} ---")
        print(f"Current dataset size: {len(X_train)}")

        X_scaled = (X_train - lb) / (ub - lb)
        gp, theta_next, pred_mean, pred_std = suggest_next(X_scaled, y_train, n_test=n_test, beta=beta)

        print(f"GP suggests theta: {theta_next}")
        print(f"GP prediction: mean={pred_mean:.4f}, std={pred_std:.4f}")

        print("Evaluating black_box(theta_next)...")
        y_next = black_box(theta_next)
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