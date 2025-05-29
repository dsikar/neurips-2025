import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from glob import glob

# Helper function to load configuration
def load_config(config_path):
    """Load JSON configuration file."""
    import json
    with open(config_path, 'r') as f:
        return json.load(f)

# Define steering values for each bin configuration
STEERING_VALUES = {
    15: [-0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005, 0.0,
         0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065],
    5: [-0.065, -0.015, 0.0, 0.015, 0.065],
    3: [-0.065, 0.0, 0.065]
}

#########
# MODEL #
#########

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs, dropout_rate=0.1):
        super(NVIDIANet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Dense layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_outputs)
        
    def forward(self, x):
        # Input normalization
        x = x / 255.0
        
        # Convolutional layers with ELU activation and dropout
        x = F.elu(self.conv1(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv2(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv3(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv4(x))
        x = self.dropout(x)
        
        x = F.elu(self.conv5(x))
        x = self.dropout(x)
        
        # Flatten and dense layers
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        
        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)
        
        return x

def load_model(model, model_path, device='cuda'):
    """Load a saved model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

class ImageProcessor:
    def __init__(self, config):
        """Initialize with preprocessing parameters from config."""
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load image processing parameters
        img_proc_config = config['image_processing']
        self.crop_top = img_proc_config['crop_top']
        self.crop_bottom = img_proc_config['crop_bottom']
        self.resize_width = img_proc_config['resize_width']
        self.resize_height = img_proc_config['resize_height']

    def preprocess_image(self, img):
        """Preprocess image for neural network."""
        # Crop and resize using config values
        cropped = img[self.crop_top:self.crop_bottom, :]
        resized = cv2.resize(cropped, (self.resize_width, self.resize_height))
        
        # Convert to YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        
        # Prepare for PyTorch
        yuv = yuv.transpose((2, 0, 1))
        yuv = np.ascontiguousarray(yuv)
        
        return torch.from_numpy(yuv).float().unsqueeze(0).to(self.device)

def get_true_label_from_filename(filename, steering_values):
    """Extract steering angle from filename and map to class index."""
    try:
        # Extract steering angle from filename (e.g., '20250523_194843_457653_steering_0.0350.jpg')
        steering_str = filename.split('_steering_')[-1].replace('.jpg', '')
        steering_value = float(steering_str)
        # Round to 3 decimal places to match steering_values
        steering_value = round(steering_value, 3)
        # Find closest steering value index
        for i, val in enumerate(steering_values):
            if abs(val - steering_value) < 1e-5:
                return i
        raise ValueError(f"Steering value {steering_value} not found in steering_values")
    except Exception as e:
        raise ValueError(f"Error parsing steering angle from {filename}: {e}")

def main(args):
    """Process dataset images and predict steering classes."""
    # Validate bins
    if args.bins not in STEERING_VALUES:
        raise ValueError(f"Invalid --bins value: {args.bins}. Must be one of {list(STEERING_VALUES.keys())}")
    
    # Load steering values
    steering_values = STEERING_VALUES[args.bins]
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NVIDIANet(num_outputs=args.bins)
    model = load_model(model, args.model, device)
    
    # Initialize image processor
    processor = ImageProcessor(config)
    
    # Load dataset images (sorted alphabetically)
    data_dir = args.data_dir
    image_files = sorted(glob(os.path.join(data_dir, '*.jpg')))
    if not image_files:
        raise FileNotFoundError(f"No .jpg files found in {data_dir}")
    
    # Initialize results array: [softmax_output (num_bins), true_label (1), predicted_label (1)]
    results = np.zeros((len(image_files), args.bins + 2))
    
    # Process each image
    for idx, image_file in enumerate(image_files):
        # Extract true label from filename
        true_label = get_true_label_from_filename(os.path.basename(image_file), steering_values)
        
        # Load and preprocess image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Failed to load image {image_file}. Skipping.")
            continue
        img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB
        
        # Preprocess for model
        processed_img = processor.preprocess_image(img)
        
        # Predict
        with torch.no_grad():
            softmax_output = model(processed_img).cpu().numpy()[0]  # Shape: (num_bins,)
            predicted_label = np.argmax(softmax_output)
        
        # Store results
        results[idx, :args.bins] = softmax_output
        results[idx, args.bins] = true_label
        results[idx, args.bins + 1] = predicted_label
        
        # Print progress
        print(f"Processed {os.path.basename(image_file)}: True Label = {true_label} ({steering_values[true_label]}), "
              f"Predicted Label = {predicted_label} ({steering_values[predicted_label]})")
    
    # Save results to .npy file
    output_file = os.path.join(data_dir, args.filename)
    np.save(output_file, results)
    print(f"Saved predictions to {output_file}")
    
    # Print summary
    correct = np.sum(results[:, args.bins] == results[:, args.bins + 1])
    accuracy = correct / len(image_files) if image_files else 0.0
    print(f"Summary: Processed {len(image_files)} images, Accuracy: {accuracy:.4f} ({int(correct)}/{len(image_files)})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict steering classes for dataset images.')
    parser.add_argument('--config', type=str, default='/home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json',
                        help='Path to the configuration JSON file (default: config.json)')
    parser.add_argument('--model', type=str, default='/home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_20250525-204108.pth',
                        help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, default='/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06',
                        help='Path to the dataset directory containing .jpg images')
    parser.add_argument('--filename', type=str, default='5_bin_softmax_outputs.npy',
                        help='Output .npy filename (default: 15_bin_softmax_outputs.npy)')
    parser.add_argument('--bins', type=int, default=5, choices=[3, 5, 15],
                        help='Number of steering bins (3, 5, or 15; default: 15)')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")

"""
# 15 bins
python 13-generate-softmax-outputs.py \
--config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_05.json \
--model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_20250523-210036.pth \
--data_dir /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05 \
--filename 15_bin_softmax_outputs.npy \
--bins 15

# 5 bins
python 13-generate-softmax-outputs.py \
--config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_06.json \
--model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_5_bins_20250525-204108.pth \
--data_dir /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06 \
--filename 5_bin_softmax_outputs.npy \
--bins 5

# For 3 bins
python script.py \
python 13-generate-softmax-outputs.py \
--config /home/daniel/git/neurips-2025/scripts/config_640x480_segmented_07_3_bins.json \
--model /home/daniel/git/neurips-2025/scripts/best_quantized_steering_model_3_bins_20250525-204246.pth \
--data_dir /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06 \
--filename 3_bin_softmax_outputs.npy \
--bins 3
"""