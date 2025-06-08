import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

def apply_noise(image, noise_type, intensity):
    """
    Apply noise to an RGB image.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be an RGB NumPy array (H, W, 3) with dtype uint8.")
    if intensity < 1 or intensity > 10:
        raise ValueError("Intensity must be between 1 and 10.")
    if noise_type not in ['brightness', 'contrast', 'defocus_blur', 'fog', 'frost', 'gaussian_noise',
                          'impulse_noise', 'motion_blur', 'pixelation', 'shot_noise', 'snow', 'zoom_blur']:
        raise ValueError("Invalid noise type.")

    result = image.copy()
    h, w = image.shape[:2]
    
    if noise_type == 'brightness':
        # Exponential scaling for aggressive brightening
        delta = 10 * (2 ** (intensity - 1) - 1)  # 0, 10, 30, 70, 150, 310, 630, 1270, etc. (capped at 255)
        result = np.clip(result.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        
    elif noise_type == 'contrast':
        factor = 1 + (intensity - 5) * 0.2
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = np.clip((result.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        
    elif noise_type == 'defocus_blur':
        kernel_size = 2 * intensity + 1
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), sigmaX=intensity)
        
    elif noise_type == 'fog':
        alpha = intensity / 20
        fog = np.full_like(result, 255, dtype=np.float32)
        result = cv2.addWeighted(result.astype(np.float32), 1 - alpha, fog, alpha, 0).astype(np.uint8)
        
    elif noise_type == 'frost':
        noise = np.random.normal(0, intensity * 5, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        result = cv2.GaussianBlur(result, (5, 5), sigmaX=intensity / 2)
        
    elif noise_type == 'gaussian_noise':
        # Exaggerated exponential scaling for Gaussian noise
        sigma = 10 * (2 ** (intensity - 1))  # 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120
        noise = np.random.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    elif noise_type == 'impulse_noise':
        prob = intensity / 100
        noise = np.random.random(result.shape[:2])
        result[noise < prob / 2] = 0
        result[noise > 1 - prob / 2] = 255
        
    elif noise_type == 'motion_blur':
        size = intensity * 2 + 1
        kernel = np.zeros((size, size))
        kernel[int(size // 2), :] = np.ones(size) / size
        result = cv2.filter2D(result, -1, kernel)
        
    elif noise_type == 'pixelation':
        scale = 1 / (intensity + 1)
        small = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
    elif noise_type == 'shot_noise':
        factor = 1 / (intensity * 0.5)
        result = np.random.poisson(result.astype(np.float32) * factor) / factor
        result = np.clip(result, 0, 255).astype(np.uint8)
        
    elif noise_type == 'snow':
        prob = intensity / 200
        noise = np.random.random(result.shape[:2])
        result[noise < prob] = 255
        
    elif noise_type == 'zoom_blur':
        result_float = result.astype(np.float32)
        for i in range(1, intensity + 1):
            scale = 1 + i * 0.05
            scaled = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled = scaled[(scaled.shape[0] - h) // 2:(scaled.shape[0] + h) // 2,
                           (scaled.shape[1] - w) // 2:(scaled.shape[1] + w) // 2]
            result_float += scaled / intensity
        result = np.clip(result_float, 0, 255).astype(np.uint8)
    
    return result

def create_noise_evaluation_grid(image, noise_type, output_dir="noise_evaluation"):
    """
    Create a grid of 11 images (original + 10 intensity levels) for a given noise type.
    Save the result with labels and title.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be an RGB NumPy array (H, W, 3) with dtype uint8.")
    if noise_type not in ['brightness', 'contrast', 'defocus_blur', 'fog', 'frost', 'gaussian_noise',
                          'impulse_noise', 'motion_blur', 'pixelation', 'shot_noise', 'snow', 'zoom_blur']:
        raise ValueError("Invalid noise type.")

    h, w = image.shape[:2]
    label_height = 50
    title_height = 50
    padding = 10

    images = [image]
    for intensity in range(1, 11):
        noisy_image = apply_noise(image, noise_type, intensity)
        images.append(noisy_image)

    grid_width = (w + padding) * 11 - padding
    grid_height = h + label_height + title_height
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    for i, img in enumerate(images):
        x_start = i * (w + padding)
        grid[title_height:title_height + h, x_start:x_start + w] = img

    grid_pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(grid_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    title = f"Noise Type: {noise_type.replace('_', ' ').title()}"
    draw.text((grid_width // 2 - 100, 10), title, fill=(0, 0, 0), font=font)

    for i in range(11):
        x_start = i * (w + padding) + w // 2 - 50
        y_start = title_height + h + 10
        if i == 0:
            label = "Original"
        else:
            label = f"Noise: {noise_type.replace('_', ' ').title()} Intensity: {i}"
        draw.text((x_start, y_start), label, fill=(0, 0, 0), font=font)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{noise_type}_evaluation.jpg")
    grid_pil.save(output_path)
    print(f"Saved evaluation grid for {noise_type} to {output_path}")

def evaluate_all_noise_types(image_path, output_dir="noise_evaluation"):
    """
    Evaluate all noise types on a single image and save grids.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    noise_types = ['brightness', 'contrast', 'defocus_blur', 'fog', 'frost', 'gaussian_noise',
                   'impulse_noise', 'motion_blur', 'pixelation', 'shot_noise', 'snow', 'zoom_blur']

    for noise_type in noise_types:
        create_noise_evaluation_grid(image, noise_type, output_dir)

if __name__ == "__main__":
    image_path = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/20250523_194843_244244_steering_0.0250.jpg" # Replace with your image path
    evaluate_all_noise_types(image_path)

# # Example usage
# if __name__ == "__main__":
#     # Replace with your image path
#     image_path = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/20250523_194843_244244_steering_0.0250.jpg"
#     evaluate_all_noise_types(image_path)