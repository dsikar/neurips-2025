import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define steering angle labels for each bin configuration
bin_labels = {
    3: [-0.065, 0.0, 0.065],
    5: [-0.065, -0.015, 0.0, 0.015, 0.065],
    15: [-0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005, 0.0, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065]
}

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs=3, dropout_rate=0.1):
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

def load_model(model_path, num_outputs, device='cuda'):
    """Load a saved model"""
    model = NVIDIANet(num_outputs=num_outputs)
    # Fix: Add weights_only=False to handle numpy objects in checkpoint
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

def steering_angle_to_class(steering_angle, steering_values):
    """Convert continuous steering angle to discrete class."""
    # Find the closest bin
    distances = [abs(steering_angle - val) for val in steering_values]
    return distances.index(min(distances))

def preprocess_image(img_path):
    """Preprocess image for neural network (same as CARLA script)"""
    # Load image
    img = np.array(Image.open(img_path).convert("RGB"))
    
    # --- CRITICAL FIX: MATCH TRAINING SCRIPT'S CROPPING ---
    crop_top = 210    # Original: 200
    crop_bottom = 480 # Original: 400
    resize_width = 200
    resize_height = 66
    
    # Crop and resize
    cropped = img[crop_top:crop_bottom, :]
    resized = cv2.resize(cropped, (resize_width, resize_height))
    
    # Convert to YUV
    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
    
    # Prepare for PyTorch
    yuv = yuv.transpose((2, 0, 1))
    yuv = np.ascontiguousarray(yuv)
    
    return torch.from_numpy(yuv).float().unsqueeze(0)

def predict_image(model, device, image_path, steering_values):
    """Predict steering class for a single image."""
    processed_img = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        class_probs = model(processed_img)
        predicted_class = torch.argmax(class_probs, dim=1).item()
    
    return predicted_class

def save_confusion_matrix(cm, class_labels, dataset_name, output_dir):
    """Save confusion matrix as image and text file."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'{label:.3f}' for label in class_labels],
                yticklabels=[f'{label:.3f}' for label in class_labels])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'{dataset_name}_confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save as text file
    text_path = os.path.join(output_dir, f'{dataset_name}_confusion_matrix.txt')
    with open(text_path, 'w') as f:
        f.write(f"Confusion Matrix - {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Class labels: " + ", ".join([f'{label:.3f}' for label in class_labels]) + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write("Rows: True classes, Columns: Predicted classes\n\n")
        
        # Write header
        f.write("True\\Pred\t")
        for label in class_labels:
            f.write(f"{label:.3f}\t")
        f.write("\n")
        
        # Write matrix
        for i, true_label in enumerate(class_labels):
            f.write(f"{true_label:.3f}\t")
            for j in range(len(class_labels)):
                f.write(f"{cm[i, j]}\t")
            f.write("\n")
    
    print(f"Confusion matrix saved to: {plot_path}")
    print(f"Confusion matrix text saved to: {text_path}")

def evaluate_model(model_path, data_dir, bins):
    """Evaluate the CNN model on all images in the data directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steering_values = bin_labels[bins]
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, num_outputs=bins, device=device)
    print(f"Model loaded successfully on device: {device}")
    
    # Get all image files and sort alphabetically
    image_files = [f for f in os.listdir(data_dir) 
                   if f.lower().endswith('.jpg') and 'steering_' in f]
    image_files.sort()
    
    print(f"Found {len(image_files)} images to evaluate")
    print(f"Using {bins} bins: {steering_values}")
    
    predictions = []
    true_classes = []
    
    for i, image_file in enumerate(image_files):
        try:
            # Extract true label from filename
            true_steering = extract_label_from_filename(image_file)
            true_class = steering_angle_to_class(true_steering, steering_values)
            
            # Get prediction
            image_path = os.path.join(data_dir, image_file)
            predicted_class = predict_image(model, device, image_path, steering_values)
            
            predictions.append(predicted_class)
            true_classes.append(true_class)
            
            # Print progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    # Calculate metrics
    if predictions and true_classes:
        accuracy = accuracy_score(true_classes, predictions)
        cm = confusion_matrix(true_classes, predictions)
        
        print(f"\nEvaluation Results:")
        print(f"Total images processed: {len(predictions)}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Print classification report
        class_names = [f'{val:.3f}' for val in steering_values]
        report = classification_report(true_classes, predictions, 
                                     target_names=class_names, 
                                     zero_division=0)
        print(f"\nClassification Report:")
        print(report)
        
        # Save confusion matrix
        dataset_name = os.path.basename(data_dir.rstrip('/'))
        output_dir = os.path.dirname(data_dir) if os.path.dirname(data_dir) else '.'
        save_confusion_matrix(cm, steering_values, dataset_name, output_dir)
        
        # Print confusion matrix
        print(f"\nConfusion Matrix:")
        print("Rows: True classes, Columns: Predicted classes")
        print("Class labels:", [f'{val:.3f}' for val in steering_values])
        print(cm)
        
        # Show some example predictions
        print(f"\nSample predictions (first 10):")
        for i in range(min(10, len(predictions))):
            true_steering = extract_label_from_filename(image_files[i])
            pred_steering = steering_values[predictions[i]]
            true_class = true_classes[i]
            pred_class = predictions[i]
            correct = "✓" if true_class == pred_class else "✗"
            print(f"{image_files[i]}: True={true_steering:.3f}(class {true_class}), "
                  f"Pred={pred_steering:.3f}(class {pred_class}) {correct}")
    
    else:
        print("No valid predictions were generated.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN Classification Model on Training Data")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained CNN model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing training data images")
    parser.add_argument("--bins", type=int, choices=[3, 5, 15], required=True,
                       help="Number of steering angle bins/classes")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    evaluate_model(args.model_path, args.data_dir, args.bins)

if __name__ == "__main__":
    main()

"""
Example usage:
python cnn-classifier-evaluation.py \
    --model_path ~/path/to/your/cnn/model.pth \
    --data_dir ~/path/to/training/data/directory \
    --bins 3
"""