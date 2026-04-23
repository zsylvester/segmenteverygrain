# Script with functions for synthetic noise modeling
from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import differential_evolution

@dataclass
class NoiseParams:
    a: float              
    b: float              
    sigma_r: float        
    l: float            
    k: float 

rng = np.random.default_rng(2) #For reproducibility with random sampling

#Input clear image -> output image with synthetic noise based on parameters theta
def synthetic_noise(x, params, rng):
    n_signal = sample_signal_dependent_noise(x, params.a, params.b, rng)
    n_row = sample_row_noise(x.shape, params.sigma_r, rng)
    n_corr = sample_correlated_noise(x.shape, params.l, params.k, rng)
    return x + n_signal + n_row + n_corr

#Synthetic Noise helper functions
def sample_signal_dependent_noise(x: np.ndarray, a: float, b: float, rng: np.random.Generator):
    """
    sqrt(a*x + b) * N(0,1) = N(0,ax+b)
    """
    var_map = np.maximum(a * x + b, 1e-12)  # avoid negative / zero
    std_map = np.sqrt(var_map)
    return rng.normal(0.0, std_map, size=x.shape).astype(np.float32)

def sample_row_noise(shape: tuple[int, int], sigma_r: float, rng: np.random.Generator) -> np.ndarray:
    """
    N(0,sigma_r^2) for each row individually
    """
    rows, columns = shape
    row_offsets = rng.normal(0.0, sigma_r, size=(rows, 1)).astype(np.float32)
    return np.repeat(row_offsets, columns, axis=1)

def sample_correlated_noise(shape: tuple[int, int], l: float, k: float, rng: np.random.Generator) -> np.ndarray:
    """
    ktimes std unit G_l convolution with N(0,1)
    """
    # N(0,1)
    random_norm = rng.normal(0.0, 1.0, size=shape).astype(np.float32)

    # apply convolution with Gaussian kernel (l = sigma)
    sigma = max(float(l), 0.01)
    correlated = cv2.GaussianBlur(random_norm, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)

    # normalize to unit std
    std = float(np.std(correlated))
    if std > 1e-12:
        correlated = correlated / std

    return (k * correlated).astype(np.float32)


#Get rough noise with high-pass filtering
def extract_noise(img,filter_strength=1.0):
    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=filter_strength, sigmaY=filter_strength)
    return img - smooth

def get_noise_stats(img,num_bins=10,pixel_cutoff=30):
    """
    Computes variances and means of noise
    """
    values = img.flatten()
    noise = extract_noise(img).flatten()

    bins = np.linspace(0, 1, num_bins + 1)
    bin_idx = np.digitize(values, bins) - 1

    var_list = []
    mean_list = []

    for i in range(num_bins):
        mask = bin_idx == i
        if np.sum(mask) > pixel_cutoff:
            var_list.append(np.var(noise[mask]))
            mean_list.append(np.mean(values[mask]))

    return np.array(mean_list), np.array(var_list)



#Input image with synthetic noise, and real noise -> output loss estimated by comparing noise stats of the two
def get_noise_loss(synthetic_img: np.ndarray, real_img: np.ndarray) -> float:
    """
    Calculates a noise loss for matching; m = mean, v = variance
    """
    #Variance loss
    m_syn, v_syn = get_noise_stats(synthetic_img)
    m_real, v_real = get_noise_stats(real_img)
    v_real_interp = np.interp(m_syn, m_real, v_real)
    loss_var = np.mean((v_syn - v_real_interp) ** 2)

    #Row noise loss
    row_syn = np.var(np.mean(synthetic_img, axis=1))
    row_real = np.var(np.mean(real_img, axis=1))

    loss_row = (row_syn - row_real) ** 2

    #Correlation loss, autocorrelation lag 1
    def autocorr(img):
        noise = extract_noise(img)
        return np.mean(noise[:, :-1] * noise[:, 1:])
    corr_syn = autocorr(synthetic_img)
    corr_real = autocorr(real_img)
    loss_corr = (corr_syn - corr_real) ** 2

    total_loss = loss_var + loss_row + loss_corr
    return float(total_loss)

import time
from datetime import timedelta

_obj_call_count = 0
_obj_start = None

def _log_obj_start(n_images):
    global _obj_call_count, _obj_start
    _obj_call_count += 1
    _obj_start = time.time()
    print(f"  OBJ#{_obj_call_count}: evaluating {n_images} images...")

def _log_obj_end(loss, n_images):
    global _obj_start
    elapsed = time.time() - _obj_start
    print(f"  OBJ#{_obj_call_count}: loss={loss:.3e} ({elapsed:.1f}s total, {elapsed/n_images:.2f}s/img)")

def _log_image_progress(i, total, img_shape):
    print(f"    img {i+1}/{total} ({img_shape[0]}x{img_shape[1]})")

# Optimize parameters to match noise statistics
def objective(theta_vec, clear_imgs, real_imgs, seed=2):
    """
    theta_vec = [a, b, sigma_r, l, k]
    Returns average noise loss across image pairs.
    """
    params = NoiseParams(
        a=float(theta_vec[0]),
        b=float(theta_vec[1]),
        sigma_r=float(theta_vec[2]),
        l=float(theta_vec[3]),
        k=float(theta_vec[4]),
    )

    rng = np.random.default_rng(seed)

    n = min(len(clear_imgs), len(real_imgs))
    _log_obj_start(n)
    losses = []

    for i in range(n):
        _log_image_progress(i, n, real_imgs[i].shape)
        clean_lowres = downsample_to_shape(clear_imgs[i], real_imgs[i].shape)

        clean_lowres = percentile_normalize(clean_lowres)
        real_img = percentile_normalize(real_imgs[i])

        syn_img = synthetic_noise(clean_lowres, params, rng)
        loss = get_noise_loss(syn_img, real_img)
        losses.append(loss)

    avg_loss = float(np.mean(losses))
    _log_obj_end(avg_loss, n)
    return avg_loss

#Image prepping
def load_images_from_folder(folder: str):
    """
    Load all readable grayscale images from a folder.
    """
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = []
    paths = []

    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img.astype(np.float32) / 255.0)
        paths.append(str(p))

    return images, paths

def downsample_to_shape(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Downsample a clean image to the shape of a lower-resolution image.
    target_shape = (rows, cols)
    """
    rows, cols = target_shape
    return cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)

def percentile_normalize(img: np.ndarray, lo=1, hi=99) -> np.ndarray:
    """
    Optional robust normalization for SEM-like images.
    """
    p_lo, p_hi = np.percentile(img, [lo, hi])
    img = (img - p_lo) / (p_hi - p_lo + 1e-8)
    return np.clip(img, 0.0, 1.0)

if __name__ == "__main__":
    clear_folder = "./testcleanimages/"
    real_folder = "./testnoisyimages/"

    print(f"Loading images from {clear_folder}...")
    clear_imgs, _ = load_images_from_folder(clear_folder)
    real_imgs, _ = load_images_from_folder(real_folder)
    print(f"Loaded {len(clear_imgs)} clear, {len(real_imgs)} real images")

bounds = [
        (1e-6, 0.2),   # a
        (1e-6, 0.05),  # b
        (1e-6, 0.05),  # sigma_r
        (0.3, 8.0),    # l
        (1e-6, 0.1),   # k
    ]

class OptimizationLogger:
    def __init__(self, maxiter):
        self.maxiter = maxiter
        self.start_time = time.time()
        self.last_log = self.start_time

    def __call__(self, xk, convergence):
        elapsed = time.time() - self.start_time
        iters = int(convergence * self.maxiter) if convergence else 0
        since_last = time.time() - self.last_log

        print(f"[{timedelta(seconds=int(elapsed))}] iter={iters}/{self.maxiter} | "
                f"a={xk[0]:.2e}, b={xk[1]:.2e}, sigma_r={xk[2]:.2e} | "
                f"l={xk[3]:.2f}, k={xk[4]:.3f} | "
                f"+{since_last:.1f}s")

        self.last_log = time.time()
        return False

logger = OptimizationLogger(maxiter=20)
    
result = differential_evolution(
    objective,
    bounds=bounds,
    args=(clear_imgs[:10], real_imgs[:10], 2),
    maxiter=20,
    popsize=10,
    polish=True,
    seed=2,
    callback=logger,
    disp=True
)

best_params = NoiseParams(
    a=float(result.x[0]),
    b=float(result.x[1]),
    sigma_r=float(result.x[2]),
    l=float(result.x[3]),
    k=float(result.x[4]),
)

print("Optimized parameters:")
print(best_params.a,best_params.b,best_params.sigma_r,best_params.l,best_params.k)
print("Best loss:", float(result.fun))