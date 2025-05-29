import os
import glob
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Define the discrete steering angles globally -3 bins
STEERING_ANGLES = [-0.065, 0.0, 0.065]

def get_carla_data_files(data_dir: str, min_timestamp: str = None) -> List[Tuple[str, float]]:
    """
    Get all valid training files from the Carla dataset directory and their steering angles.
    
    Args:
        data_dir: Path to the carla_dataset directory
        min_timestamp: Minimum timestamp to include (as string), optional
                      If None, includes all valid files
    
    Returns:
        List of tuples containing (file_path, steering_angle)
    """
    pattern = os.path.join(data_dir, "*.jpg")
    all_files = glob.glob(pattern)
    valid_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        if len(parts) >= 5 and 'steering' in filename:
            timestamp = '_'.join(parts[0:3])
            try:
                steering = float(parts[-1].replace('.jpg', ''))
                if min_timestamp is None or timestamp >= min_timestamp:
                    valid_files.append((file_path, steering))
            except ValueError:
                continue
    
    valid_files.sort(key=lambda x: os.path.basename(x[0]).split('_')[0:3])
    return valid_files

def train_val_split(file_pairs: List[Tuple[str, float]], val_ratio: float = 0.2) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Split the dataset into training and validation sets.
    
    Args:
        file_pairs: List of (file_path, steering_angle) tuples
        val_ratio: Ratio of validation set size to total dataset size
    
    Returns:
        Tuple of (train_pairs, val_pairs)
    """
    num_samples = len(file_pairs)
    indices = np.random.permutation(num_samples)
    split_idx = int(np.floor(val_ratio * num_samples))
    
    val_indices = indices[:split_idx]
    train_indices = indices[split_idx:]
    
    train_pairs = [file_pairs[i] for i in train_indices]
    val_pairs = [file_pairs[i] for i in val_indices]
    
    return train_pairs, val_pairs

class CarlaDataset(Dataset):
    """
    PyTorch Dataset for Carla steering angle classification.
    Handles loading and preprocessing of images, and conversion to class indices.
    """
    def __init__(self, 
                 file_pairs: List[Tuple[str, float]], 
                 crop_top: int = 210,
                 crop_bottom: int = 480,
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            file_pairs: List of tuples containing (image_path, steering_angle)
            crop_top: Y coordinate where crop begins
            crop_bottom: Y coordinate where crop ends
            transform: Optional additional transformations
        """
        self.file_pairs = file_pairs
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.transform = transform
        self.steering_angles = STEERING_ANGLES

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.file_pairs)
    
    def prepare_image_for_neural_network(self, image_path: str) -> np.ndarray:
        """
        Load image, crop, resize, and convert to YUV for neural network processing.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            numpy array in YUV format, size 66x200x3
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = img_rgb[self.crop_top:self.crop_bottom, :]
        resized = cv2.resize(cropped, (200, 66))
        yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
        return yuv
    
    def steering_to_class(self, steering: float) -> int:
        """
        Convert a continuous steering angle to a class index based on the nearest discrete angle.
        
        Args:
            steering: Continuous steering angle (float)
            
        Returns:
            Class index (int) corresponding to the nearest discrete angle
        """
        if steering < self.steering_angles[0]:
            return 0
        elif steering > self.steering_angles[-1]:
            return len(self.steering_angles) - 1
        else:
            return min(range(len(self.steering_angles)), key=lambda i: abs(self.steering_angles[i] - steering))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            tuple: (image, class_index) where image is a preprocessed torch tensor
                  and class_index is a torch tensor (long)
        """
        image_path, steering_angle = self.file_pairs[idx]
        
        image = self.prepare_image_for_neural_network(image_path)
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
        
        # Convert steering angle to class index
        class_index = self.steering_to_class(steering_angle)
        class_tensor = torch.tensor(class_index, dtype=torch.long)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, class_tensor

class NVIDIANet(nn.Module):
    def __init__(self, num_outputs=3, dropout_rate=0.1):  # Changed to 3 outputs
        super(NVIDIANet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, num_outputs)  # Outputs 3 logits for classification

    def forward(self, x):
        x = x / 255.0
        
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
        
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)  # Output logits (no activation)
        
        return x

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=6, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict().copy()
            self.counter = 0
            return True
        return False

def train_model(model, train_loader, val_loader, model_save_path, num_epochs=100, device="cuda", learning_rate=1e-5):
    """Training loop with validation and early stopping"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # Changed to cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=6)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],  # NEW
        'val_acc': []     # NEW
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_correct = 0  # NEW
        train_total = 0    # NEW
                
        for images, class_indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            class_indices = class_indices.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, class_indices)  # No squeeze, expects logits and class indices
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())

                        # NEW: Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += class_indices.size(0)
            train_correct += (predicted == class_indices).sum().item()

        # NEW: Store training accuracy
        avg_train_acc = train_correct / train_total
        history['train_acc'].append(avg_train_acc)            
        
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, class_indices in val_loader:
                images = images.to(device)
                class_indices = class_indices.to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, class_indices)
                val_losses.append(val_loss.item())


                # NEW: Validation accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += class_indices.size(0)
                val_correct += (predicted == class_indices).sum().item()                
        
        # NEW: Store validation accuracy
        avg_val_acc = val_correct / val_total
        history['val_acc'].append(avg_val_acc)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Train Acc: {avg_train_acc:.2%}, '  # NEW
              f'Val Acc: {avg_val_acc:.2%}')        # NEW
                
        is_best = early_stopping(avg_val_loss, model)
        
        if is_best:
            print(f"New best model! Saving to {model_save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_save_path)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            print(f"Loading best model from {model_save_path}")
            model.load_state_dict(early_stopping.best_state_dict)
            break
    
    return model, history

def load_model(model, model_path, device='cuda'):
    """Load a saved model"""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def plot_training_history(history):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('training_history.png')

def train_steering_model(train_dataset, val_dataset, model_save_path, batch_size=64):
    """Full training pipeline"""
    start_time = datetime.datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = NVIDIANet(num_outputs=3)  # Updated to 3 outputs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_save_path=model_save_path,
        num_epochs=100,
        device=device,
        learning_rate=1e-5
    )

    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print(f"Training ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {str(execution_time)}")

    model_name = os.path.basename(model_save_path).replace('.pth', '')
    log_file = f"{model_name}_training.log"
    with open(log_file, 'w') as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Training Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution Time: {str(execution_time)}\n")

    plot_training_history(history)
    
    return trained_model, history

# Example usage in Jupyter notebook:
data_dir = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/" # segmentation, no fiducial markers
file_pairs = get_carla_data_files(data_dir)
print(f"Total number of samples: {len(file_pairs)}")

train_pairs, val_pairs = train_val_split(file_pairs)
print(f"Training samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}")

train_dataset = CarlaDataset(train_pairs)
val_dataset = CarlaDataset(val_pairs)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

images, class_indices = next(iter(train_loader))
print(f"Batch image shape: {images.shape}")  # [batch_size, 3, 66, 200]
print(f"Batch class indices shape: {class_indices.shape}")  # [batch_size]

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = f'best_quantized_steering_model_3_bins_balanced_{timestamp}.pth'

model, history = train_steering_model(
    train_dataset, 
    val_dataset, 
    model_save_path=model_save_path
)

# For inference:
model = NVIDIANet(num_outputs=3)
model = load_model(model, model_save_path)

# Example inference function to convert class index to steering angle
def predict_steering_angle(model, image_tensor, device='cuda'):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        if len(image_tensor.shape) == 3:  # Add batch dimension if needed
            image_tensor = image_tensor.unsqueeze(0)
        logits = model(image_tensor)
        class_index = torch.argmax(logits, dim=1).item()
        steering_angle = STEERING_ANGLES[class_index]
    return steering_angle