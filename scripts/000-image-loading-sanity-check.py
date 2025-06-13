# check if image was saved in RGB or BRG
# "Carla-Cola" sign should be RED if saved in RGB, otherwise blue-ish
import cv2
import numpy as np
import matplotlib.pyplot as plt

def inspect_image(image_path):
    """
    Read an image, inspect its color space, and plot it with Matplotlib.
    
    Args:
        image_path (str): Path to the image file.
    """
    # Read the image with OpenCV (loads in BGR format)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # Print image shape and pixel value at fiducial marker location
    print(f"Image shape: {img.shape}")  # Expected: (480, 640, 3)
    marker_pixel = img[470, 100]  # Left fiducial marker (x=90, y=460)
    print(f"Pixel value at (x=90, y=460): {marker_pixel}")
    # In BGR: Red marker should be ~[0, 0, 255]
    # In RGB: Red marker should be ~[255, 0, 0]
    
    # Convert BGR to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Plot the image
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title("Image with Fiducial Markers and Waypoint Lines")
    plt.axis('off')  # Hide axes
    plt.show()

if __name__ == "__main__":
    # Path to the image
    image_path = "carla_dataset_640x480_segmented_with_fiducials/20250512_205810_792747_steering_0.0000.jpg"
    
    try:
        inspect_image(image_path)
    except Exception as e:
        print(f"Error: {e}")
