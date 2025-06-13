# Model evaluation script for steering angle prediction
# Evaluates a trained CNN model on a dataset of images

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import re
import argparse
from config_utils import load_config

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

def load_model(model_path, num_outputs=1, device='cuda'):
    """Load a saved model"""
    model = NVIDIANet(num_outputs=num_outputs)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def extract_label_from_filename(filename):
    """Extract steering angle label from filename."""
    match = re.search(r"steering_([-0-9.]+)\.jpg", filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    return float(match.group(1))

class ModelEvaluator:
    def __init__(self, config, model_path, dataset_path):
        self.config = config
        self.dataset_path = dataset_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = load_model(model_path, num_outputs=1, device=self.device)
        
        # Load image processing parameters from config
        img_proc_config = config['image_processing']
        self.crop_top = img_proc_config['crop_top']
        self.crop_bottom = img_proc_config['crop_bottom']
        self.resize_width = img_proc_config['resize_width']
        self.resize_height = img_proc_config['resize_height']
        
        # Load control parameters
        control_config = config.get('control', {})
        self.max_steering_angle = control_config.get('max_steering_angle', 1.0)
        
    def preprocess_image(self, img):
        """Preprocess image for neural network"""
        # Crop and resize using config values
        cropped = img[self.crop_top:self.crop_bottom, :]
        resized = cv2.resize(cropped, (self.resize_width, self.resize_height))
        
        # Convert to YUV
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        
        # Prepare for PyTorch
        yuv = yuv.transpose((2, 0, 1))
        yuv = np.ascontiguousarray(yuv)
        
        return torch.from_numpy(yuv).float().unsqueeze(0).to(self.device)
    
    def predict_steering(self, image):
        """Make steering prediction from image"""
        with torch.no_grad():
            steering_pred = self.model(image)
            
        steering_angle = float(steering_pred.cpu().numpy()[0, 0])
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        return steering_angle
    
    def evaluate_dataset(self):
        """Evaluate model on the entire dataset"""
        image_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.jpg')]
        
        if not image_files:
            raise ValueError(f"No .jpg files found in {self.dataset_path}")
        
        predictions = []
        ground_truths = []
        processed_count = 0
        
        print(f"Found {len(image_files)} images to process...")
        
        for i, filename in enumerate(image_files):
            try:
                # Extract ground truth label from filename
                ground_truth = extract_label_from_filename(filename)
                
                # Load and preprocess image
                img_path = os.path.join(self.dataset_path, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not load image {filename}")
                    continue
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Preprocess and predict
                processed_img = self.preprocess_image(img)
                prediction = self.predict_steering(processed_img)
                
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                processed_count += 1
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images...")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No images were successfully processed")
        
        # Calculate MAE
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)
        mae = np.mean(np.abs(predictions - ground_truths))
        
        # Calculate accuracy (within a tolerance)
        tolerance = 0.1  # Consider predictions within 0.1 steering angle as correct
        correct_predictions = np.sum(np.abs(predictions - ground_truths) <= tolerance)
        accuracy = correct_predictions / len(predictions)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Total images processed: {processed_count}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
        print(f"Mean Absolute Error: {mae:.6f}")
        
        return mae, accuracy, processed_count, correct_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate steering angle prediction model.')
    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to the configuration JSON file (default: config.json)')
    parser.add_argument('--model', type=str, default='model.pth', 
                        help='Path to the trained model file (default: model.pth)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset directory containing images')
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        evaluator = ModelEvaluator(config, args.model, args.dataset)
        evaluator.evaluate_dataset()
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
"""
CUDA_VISIBLE_DEVICES=7 python /users/aczd097/git/neurips-2025/scripts/19-generate-omae-cnn-regressors.py \
 --config /users/aczd097/git/neurips-2025/scripts/config_640x480_segmented_01.json \
 --model /users/aczd097/archive/git/neurips-2025/scripts/best_steering_model_20250514-122921.pth \
 --dataset /users/aczd097/archive/git/neurips-2025/scripts/carla_dataset_640x480_02
print("Finished processing RegCNNContUnbalanced")

#RegCNNContUnbalancedFiducials
CUDA_VISIBLE_DEVICES=7 python /users/aczd097/git/neurips-2025/scripts/19-generate-omae-cnn-regressors.py \
 --config /users/aczd097/git/neurips-2025/scripts/config_640x480_segmented_01.json \
 --model /users/aczd097/archive/git/neurips-2025/scripts/best_steering_model_20250513-093008.pth \
 --dataset /users/aczd097/archive/git/neurips-2025/scripts/carla_dataset_640x480_01
print("Finished processing RegCNNContUnbalancedFiducials")

# local
python 19-generate-omae-cnn-regressors.py \
 --config config_640x480_segmented_01.json \
 --model best_steering_model_20250513-093008.pth \
 --dataset carla_dataset_640x480_01
"""