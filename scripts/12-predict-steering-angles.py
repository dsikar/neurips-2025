import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
from pathlib import Path

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs=1, dropout_rate=0.1):
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
        
        return x

def load_model(model, model_path, device='cuda'):
    """Load a saved model"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}")

def preprocess_image(img, config):
    """Preprocess image for neural network using config parameters"""
    # Extract preprocessing parameters
    try:
        camera_config = config['camera']
        img_proc_config = config['image_processing']
        
        image_height = camera_config['image_height']
        image_width = camera_config['image_width']
        crop_top = img_proc_config['crop_top']
        crop_bottom = img_proc_config['crop_bottom']
        resize_width = img_proc_config['resize_width']
        resize_height = img_proc_config['resize_height']
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e}")
    
    # Verify image dimensions
    if img.shape[0] != image_height or img.shape[1] != image_width:
        print(f"Warning: Image dimensions {img.shape[0]}x{img.shape[1]} do not match "
              f"config {image_height}x{image_width}. Resizing to match config.")
        img = cv2.resize(img, (image_width, image_height))
    
    # Crop and resize
    cropped = img[crop_top:crop_bottom, :]
    resized = cv2.resize(cropped, (resize_width, resize_height))
    
    # Convert to YUV
    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
    
    # Prepare for PyTorch
    yuv = yuv.transpose((2, 0, 1))
    yuv = np.ascontiguousarray(yuv)
    
    return torch.from_numpy(yuv).float().unsqueeze(0)

def predict_steering_angles(input_dir, model_path, output_dir, file_prefix, config_path):
    """
    Predict steering angles for all .jpg files in input_dir and save to output_dir.
    
    Args:
        input_dir (str): Directory containing .jpg files
        model_path (str): Path to the trained model file
        output_dir (str): Directory to save output files
        file_prefix (str): Prefix for output file names
        config_path (str): Path to the configuration JSON file
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Ensure model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist")
    
    # Ensure config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    config = load_config(config_path)
    
    # Get list of .jpg files and sort alphabetically
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    jpg_files.sort()
    
    if not jpg_files:
        print("No .jpg files found in the input directory")
        return
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = NVIDIANet()
    model = load_model(model, model_path, device)
    
    # Predict steering angles
    steering_angles = []
    for file_name in jpg_files:
        try:
            # Load and preprocess image
            img_path = os.path.join(input_dir, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {file_name}")
                continue
            
            # Convert BGR to RGB
            img = img[:, :, [2, 1, 0]]
            
            # Preprocess image using config parameters
            processed_img = preprocess_image(img, config).to(device)
            
            # Predict steering angle
            with torch.no_grad():
                steering_pred = model(processed_img)
            
            steering_angle = float(steering_pred.cpu().numpy()[0, 0])
            steering_angles.append(steering_angle)
            
        except Exception as e:
            print(f"Warning: Error processing {file_name}: {e}")
            continue
    
    if not steering_angles:
        print("No steering angles predicted")
        return
    
    # Convert to numpy array
    angles_array = np.array(steering_angles)
    
    # Define output file paths
    txt_path = os.path.join(output_dir, f"{file_prefix}_steering_predictions.txt")
    npy_path = os.path.join(output_dir, f"{file_prefix}_steering_predictions.npy")
    
    # Save to text file
    with open(txt_path, 'w') as f:
        for angle in steering_angles:
            f.write(f"{angle:.4f}\n")
    
    # Save to .npy file
    np.save(npy_path, angles_array)
    
    print(f"Saved {len(steering_angles)} steering angle predictions to:")
    print(f"- Text file: {txt_path}")
    print(f"- Numpy file: {npy_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict steering angles for .jpg files')
    parser.add_argument('--input-dir', required=True, help='Input directory containing .jpg files')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--output-dir', required=True, help='Output directory for saving predictions')
    parser.add_argument('--file-prefix', required=True, help='Prefix for output file names')
    parser.add_argument('--config', required=True, help='Path to the configuration JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the prediction function
    predict_steering_angles(args.input_dir, args.model, args.output_dir, args.file_prefix, args.config)

if __name__ == "__main__":
    main()

    """
    # Predict steering angles from dataset
    # Example usage:
    python 12-predict-steering-angles.py \
        --input-dir carla_dataset_640x480_01 \
        --model best_steering_model_20250513-093008.pth\
        --output-dir steering_angles_output \
        --file-prefix best_steering_model_20250513-093008_seg_fid_predictions \
        --config config_640x480_segmented_01.json
    """