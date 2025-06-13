import numpy as np

def compute_path_error(ground_truth_file, self_driving_file):
    """Compute MAE between ground truth and self-driving path distances."""
    # Read distances
    with open(ground_truth_file, 'r') as f:
        ground_truth = [float(line.strip()) for line in f]
    with open(self_driving_file, 'r') as f:
        self_driving = [float(line.strip()) for line in f]
    
    # Ensure same length (truncate to shorter)
    min_length = min(len(ground_truth), len(self_driving))
    ground_truth = ground_truth[:min_length]
    self_driving = self_driving[:min_length]
    
    # Compute MAE
    mae = np.mean(np.abs(np.array(self_driving) - np.array(ground_truth)))
    print(f"Path Distance MAE: {mae:.4f} meters")
    
    # Optional: Compute MSE
    mse = np.mean((np.array(self_driving) - np.array(ground_truth))**2)
    print(f"Path Distance MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f} meters")
    
    return mae

# Example usage
# ground_truth_file = 'carla_dataset_640x480_segmented_with_fiducials/ground_truth_distances.txt'
# self_driving_file = 'self_driving_output/self_driving_distances.txt'
# mae = compute_path_error(ground_truth_file, self_driving_file)

print("Run 01, CNN x Ground Truth, segmented with fiducials")
ground_truth_file = 'carla_dataset_640x480_01/ground_truth_distances.txt'
self_driving_file = 'carla_dataset_640x480_01/self_driving_distances_01.txt'
mae = compute_path_error(ground_truth_file, self_driving_file)
# Run 01, CNN x Ground Truth, segmented with fiducials
# Path Distance MAE: 0.0295 meters
# Path Distance MSE: 0.0026, RMSE: 0.0514 meters

print("Run 02, CNN x Ground Truth, segmented without fiducials")
ground_truth_file = 'carla_dataset_640x480_02/ground_truth_distances_02.txt'
self_driving_file = 'carla_dataset_640x480_02/self_driving_distances_02.txt'
mae = compute_path_error(ground_truth_file, self_driving_file)