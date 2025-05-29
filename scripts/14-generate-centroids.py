import numpy as np
import argparse
import os

def compute_centroids(softmax_file, num_bins):
    """Compute centroids (average softmax outputs) for each class from correct predictions."""
    # Load softmax outputs
    data = np.load(softmax_file)  # Shape: (num_images, num_bins + 2)
    
    # Extract columns
    softmax_outputs = data[:, :num_bins]  # Shape: (num_images, num_bins)
    true_labels = data[:, num_bins].astype(int)  # True class indices
    pred_labels = data[:, num_bins + 1].astype(int)  # Predicted class indices
    
    # Identify correct predictions (where true_label == pred_label)
    correct_mask = true_labels == pred_labels
    if not np.any(correct_mask):
        raise ValueError("No correct predictions found in the data.")
    
    # Initialize centroids array
    centroids = np.zeros((num_bins, num_bins))  # Shape: (num_bins, num_bins)
    counts = np.zeros(num_bins, dtype=int)  # Count of correct predictions per class
    
    # Compute sum of softmax outputs for correct predictions
    for i in range(len(data)):
        if correct_mask[i]:
            class_idx = true_labels[i]
            centroids[class_idx] += softmax_outputs[i]
            counts[class_idx] += 1
    
    # Compute average (centroid) for each class
    for i in range(num_bins):
        if counts[i] > 0:
            centroids[i] /= counts[i]
        else:
            print(f"Warning: No correct predictions for class {i}. Centroid set to zeros.")
    
    return centroids

def main(args):
    """Load softmax outputs, compute centroids, and save to file."""
    # Validate bins
    if args.bins not in [3, 5, 15]:
        raise ValueError(f"Invalid --bins value: {args.bins}. Must be one of [3, 5, 15]")
    
    # Compute centroids
    centroids = compute_centroids(args.filename, args.bins)
    
    # Generate output filename
    output_file = os.path.join(os.path.dirname(args.filename), f"{args.bins}_bins_centroids.npy")
    
    # Save centroids
    np.save(output_file, centroids)
    print(f"Saved centroids to {output_file}")
    
    # Print summary
    print(f"Centroids for {args.bins} bins:")
    for i, centroid in enumerate(centroids):
        print(f"Class {i}: {centroid}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute centroids from softmax outputs for correct predictions.')
    parser.add_argument('--filename', type=str, default='/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bin_softmax_outputs.npy',
                        help='Path to the softmax outputs .npy file (default: 15_bin_softmax_outputs.npy)')
    parser.add_argument('--bins', type=int, default=15, choices=[3, 5, 15],
                        help='Number of steering bins (3, 5, or 15; default: 15)')
    
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")

"""
Examples:

# 15 bins
python 14-generate-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bin_softmax_outputs.npy\
    --bins 15

# 5 bins
python 14-generate-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bin_softmax_outputs.npy\
    --bins 5

# 3 bins
python 14-generate-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bin_softmax_outputs.npy\
    --bins 3


"""        