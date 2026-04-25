from synthetic_noise import make_noisy_training_pair, NoiseParams
import numpy as np
import cv2
from pathlib import Path
from synthetic_noise import load_images_from_folder

def load_image_mask_pairs(folder: str):
    """
    Load images and masks from a single folder.
    Images contain 'image' in filename (or don't contain 'mask').
    Masks contain 'mask' in filename.
    Pairs are matched by removing '_mask' suffix from mask filename.
    """
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    
    images = {}
    masks = {}
    
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem
        if "mask" in stem.lower():
            base_name = stem.replace("_mask", "").replace("mask", "")
            masks[base_name] = str(p)
        else:
            base_name = stem.replace("_image", "").replace("image", "")
            images[base_name] = str(p)
    
    pairs = []
    for base_name, img_path in images.items():
        if base_name in masks:
            pairs.append((img_path, masks[base_name]))
    
    return pairs

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img.astype(np.float32) / 255.0

input_folder = "./cleanimages/"
output_folder = "./noisy_output/"
noise_reference_folder = "./testnoisyimages/"

pairs = load_image_mask_pairs(input_folder)
print(f"Found {len(pairs)} image-mask pairs")

_, noise_paths = load_images_from_folder(noise_reference_folder)
sample_noisy = cv2.imread(noise_paths[0], cv2.IMREAD_GRAYSCALE)
target_shape = sample_noisy.shape
print(f"Using target_shape from {noise_reference_folder}: {target_shape}")

params = NoiseParams(a=0.0023589515117326183, b=0.001712502743955444, sigma_r=0.0006997093027690107, l=0.7603779994083678, k=0.07404317063233228)
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
        rng=rng
    )
    
    noisy_path = Path(output_folder) / f"{img_name}noisy_image.png"
    mask_path_out = Path(output_folder) / f"{img_name}noisy_mask.png"
    
    cv2.imwrite(str(noisy_path), (noisy_img * 255).astype(np.uint8))
    cv2.imwrite(str(mask_path_out), (mask_ds * 255).astype(np.uint8))
    print(f"Saved: {noisy_path}")