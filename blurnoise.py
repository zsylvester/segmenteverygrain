#thanks chat
import cv2
import numpy as np
import os

def augment_image(input_dir, output_dir, blur_ksize, noise_sigma):
    """
    Apply Gaussian blur and add grayscale Gaussian noise to an image.
    
    Args:
        image_path (str): Path to input image.
        output_path (str): Path to save augmented image.
        blur_ksize (int): Kernel size for Gaussian blur (must be odd).
        noise_sigma (float): Standard deviation of Gaussian noise.
    """
    for fname in os.listdir(input_dir):
        # Load image
        image_path=os.path.join(input_dir,fname)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image {image_path}")

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

        # Generate grayscale Gaussian noise
        noise_gray = np.random.normal(0, noise_sigma, (image.shape[0], image.shape[1], 1)).astype(np.float32)
        
        noise = np.repeat(noise_gray, 3, axis=2)  # replicate for 3 channels

        # Add noise to blurred image
        noisy_blurred = cv2.add(blurred.astype(np.float32), noise)

        # Clip values to valid range [0,255] and convert back to uint8
        noisy_blurred = np.clip(noisy_blurred, 0, 255).astype(np.uint8)
        output_path=os.path.join(output_dir, fname)

        # Save result
        cv2.imwrite(output_path, noisy_blurred)

# Example usage
if __name__ == "__main__":
    augment_image("cropped","blurred", blur_ksize=17, noise_sigma=35)
