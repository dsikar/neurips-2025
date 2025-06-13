import numpy as np
import argparse
import os

def compute_distances(softmax_file, centroids_file, num_bins):
    """Compute distances from softmax outputs to true and predicted class centroids."""
    # Load softmax outputs and centroids
    softmax_data = np.load(softmax_file)  # Shape: (num_images, num_bins + 2)
    centroids = np.load(centroids_file)  # Shape: (num_bins, num_bins)
    
    # Validate shapes
    if softmax_data.shape[1] != num_bins + 2:
        raise ValueError(f"Softmax file has {softmax_data.shape[1]} columns, expected {num_bins + 2}")
    if centroids.shape != (num_bins, num_bins):
        raise ValueError(f"Centroids file has shape {centroids.shape}, expected ({num_bins}, {num_bins})")
    
    # Extract columns
    softmax_outputs = softmax_data[:, :num_bins]  # Shape: (num_images, num_bins)
    true_labels = softmax_data[:, num_bins].astype(int)  # True class indices
    pred_labels = softmax_data[:, num_bins + 1].astype(int)  # Predicted class indices
    
    # Initialize output array with two additional columns for distances
    output_data = np.zeros((softmax_data.shape[0], softmax_data.shape[1] + 2))
    output_data[:, :softmax_data.shape[1]] = softmax_data  # Copy original data
    
    # Compute distances for each row
    for i in range(softmax_data.shape[0]):
        # Get softmax output, true class, and predicted class
        softmax = softmax_outputs[i]
        true_class = true_labels[i]
        pred_class = pred_labels[i]
        
        # Compute Euclidean distance to true class centroid
        true_centroid = centroids[true_class]
        dist_to_true = np.sqrt(np.sum((softmax - true_centroid) ** 2))
        
        # Compute Euclidean distance to predicted class centroid
        pred_centroid = centroids[pred_class]
        dist_to_pred = np.sqrt(np.sum((softmax - pred_centroid) ** 2))
        
        # Store distances
        output_data[i, num_bins + 2] = dist_to_true
        output_data[i, num_bins + 3] = dist_to_pred
    
    return output_data

import numpy as np
import argparse
import os

def main(args):
    """Load softmax outputs and centroids, compute distances, and save to file."""
    # Validate bins
    if args.bins not in [3, 5, 15]:
        raise ValueError(f"Invalid --bins value: {args.bins}. Must be one of [3, 5, 15]")
    
    # Compute distances
    output_data = compute_distances(args.filename, args.centroids, args.bins)
    
    # Generate output filename
    output_dir = os.path.dirname(args.filename)
    output_file = os.path.join(output_dir, 
                              f"{args.bins}_bins_{args.network}_softmax_dist_{args.balanced}.npy")
    
    # Save output
    np.save(output_file, output_data)
    print(f"Saved output with distances to {output_file}")
    
    # Print summary
    print(f"Processed {output_data.shape[0]} images with {args.bins} bins")
    print(f"Output shape: {output_data.shape}")
    print(f"Columns: {args.bins} softmax outputs, true label, predicted label, "
          f"distance to true centroid, distance to predicted centroid")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute distances from softmax outputs to true and predicted class centroids.')
    parser.add_argument('--filename', type=str, 
                        default='/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bin_softmax_outputs.npy',
                        help='Path to the softmax outputs .npy file (default: 3_bin_softmax_outputs.npy)')
    parser.add_argument('--centroids', type=str, 
                        default='/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_cnn_centroids_unbalanced.npy',
                        help='Path to the centroids .npy file (default: 3_bins_cnn_centroids_unbalanced.npy)')
    parser.add_argument('--bins', type=int, default=3, choices=[3, 5, 15],
                        help='Number of steering bins (3, 5, or 15; default: 3)')
    parser.add_argument('--network', type=str, default='cnn',
                        help='Network architecture used (default: cnn)')
    parser.add_argument('--balanced', type=str, default='unbalanced', 
                        choices=['balanced', 'unbalanced'],
                        help='Whether the dataset is balanced or unbalanced (default: unbalanced)')
    
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")

"""
Examples:

# CNN
# unbalanced 
# 15 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bin_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_cnn_centroids_unbalanced.npy \
    --bins 15 \
    --network cnn

# 5 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bin_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_cnn_centroids_unbalanced.npy \
    --bins 5 \
    --network cnn

# 3 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bin_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_cnn_centroids_unbalanced.npy \
    --bins 3 \
    --network cnn

############
# balanced #
############

# 15 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bin_balanced_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_cnn_centroids_balanced.npy \
    --bins 15 \
    --network cnn \
    --balanced balanced

# 5 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bin_balanced_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy \
    --bins 5 \
    --network cnn \
    --balanced balanced

# 3 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bin_balanced_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_cnn_centroids_balanced.npy \
    --bins 3 \
    --network cnn \
    --balanced balanced

# ViT 
# unbalanced
# 15 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bin_vit_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_vit_centroids_unbalanced.npy \
    --bins 15 \
    --network vit
# 5 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bin_vit_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_vit_centroids_unbalanced.npy \
    --bins 5 \
    --network vit
# 3 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bin_vit_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_vit_centroids_unbalanced.npy \
    --bins 3 \
    --network vit

# balanced
# 15 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bin_vit_softmax_outputs_balanced.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_vit_centroids_balanced.npy \
    --bins 15 \
    --network vit \
    --balanced balanced
# 5 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bin_vit_softmax_outputs_balanced.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy \
    --bins 5 \
    --network vit \
    --balanced balanced
# 3 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bin_vit_softmax_outputs_balanced.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy \
    --bins 3 \
    --network vit \
    --balanced balanced
"""