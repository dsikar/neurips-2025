import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math

# Define steering angle labels for each bin configuration
bin_labels = {
    3: [-0.065, 0.0, 0.065],
    5: [-0.065, -0.015, 0.0, 0.015, 0.065],
    15: [-0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005, 0.0, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065]
}

def calculate_entropy(softmax_probs):
    """Calculate Shannon entropy for a set of softmax probabilities."""
    # Avoid log(0) by setting small probabilities to a minimum value
    probs = np.clip(softmax_probs, 1e-10, 1.0)
    return -np.sum(probs * np.log2(probs))

def plot_entropy_charts(args):
    # Validation checks
    if not os.path.exists(args.filename):
        raise FileNotFoundError(f"Input file not found: {args.filename}")
    if args.bins not in bin_labels:
        raise ValueError(f"Invalid --bins value: {args.bins}. Must be one of {list(bin_labels.keys())}")
    
    # Load the data
    data = np.load(args.filename)  # Shape: (num_images, num_bins + 4)
    num_bins = args.bins
    true_labels = data[:, num_bins].astype(int)  # True labels
    pred_labels = data[:, num_bins + 1].astype(int)  # Predicted labels
    
    # Validate data shape
    expected_cols = num_bins + 4
    if data.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, but got {data.shape[1]} in {args.filename}")
    
    # Calculate entropy for each row
    entropies = np.apply_along_axis(calculate_entropy, 1, data[:, :num_bins])
    
    # Separate correct and incorrect predictions
    correct_mask = true_labels == pred_labels
    incorrect_mask = ~correct_mask
    correct_entropies = entropies[correct_mask]
    incorrect_entropies = entropies[incorrect_mask]
    correct_labels = true_labels[correct_mask]
    incorrect_labels = true_labels[incorrect_mask]
    
    # Calculate average entropy and standard deviation per class
    correct_entropy_per_class = np.array([np.mean(correct_entropies[correct_labels == i]) 
                                         if np.any(correct_labels == i) else np.nan 
                                         for i in range(num_bins)])
    incorrect_entropy_per_class = np.array([np.mean(incorrect_entropies[incorrect_labels == i]) 
                                           if np.any(incorrect_labels == i) else np.nan 
                                           for i in range(num_bins)])
    correct_std_per_class = np.array([np.std(correct_entropies[correct_labels == i]) 
                                     if np.any(correct_labels == i) else np.nan 
                                     for i in range(num_bins)])
    incorrect_std_per_class = np.array([np.std(incorrect_entropies[incorrect_labels == i]) 
                                       if np.any(incorrect_labels == i) else np.nan 
                                       for i in range(num_bins)])
    
    # Create a figure with two rows for correct and incorrect predictions
    fig, axs = plt.subplots(2, num_bins, figsize=(num_bins * 2.5, 6))
    fig.suptitle(f'{num_bins} bin {args.balanced} {args.network.upper()} - Average Entropy for Correct and Incorrect Steering Angle Predictions', fontsize=16)
    
    # Plot correct predictions
    for i in range(num_bins):
        if not np.isnan(correct_entropy_per_class[i]):
            axs[0, i].bar(0, correct_entropy_per_class[i], color='skyblue', yerr=correct_std_per_class[i], capsize=5)
            axs[0, i].set_ylim(0, max(np.nanmax(correct_entropy_per_class), np.nanmax(incorrect_entropy_per_class)) * 1.2)
            axs[0, i].set_title(f'Steering {bin_labels[num_bins][i]:.3f} (Correct)', fontsize=10)
            axs[0, i].set_xticks([])
            axs[0, i].grid(True, axis='y', linestyle='--', alpha=0.7)
        else:
            axs[0, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axs[0, i].set_xticks([])
    
    # Plot incorrect predictions
    for i in range(num_bins):
        if not np.isnan(incorrect_entropy_per_class[i]):
            axs[1, i].bar(0, incorrect_entropy_per_class[i], color='lightcoral', yerr=incorrect_std_per_class[i], capsize=5)
            axs[1, i].set_ylim(0, max(np.nanmax(correct_entropy_per_class), np.nanmax(incorrect_entropy_per_class)) * 1.2)
            axs[1, i].set_title(f'Steering {bin_labels[num_bins][i]:.3f} (Incorrect)', fontsize=10)
            axs[1, i].set_xticks([])
            axs[1, i].grid(True, axis='y', linestyle='--', alpha=0.7)
        else:
            axs[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axs[1, i].set_xticks([])
    
    # Set x-axis label at the bottom of the figure
    fig.text(0.5, 0.04, 'Steering Angle Index', ha='center', fontsize=12)
    # Set y-axis label on the left side of the figure
    fig.text(0.04, 0.5, 'Average Entropy (bits)', va='center', rotation='vertical', fontsize=12)
    
    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.9)  # Adjust the top spacing for the main title
    
    # Generate output directory and filename for PNG
    base_dir = os.path.dirname(os.path.dirname(args.filename))  # Go up to the parent directory
    output_dir = os.path.join(base_dir, 'entropy_results', 'plots')
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    png_base_name = os.path.basename(args.filename).replace('softmax_dist', 'entropy_plot')
    png_output_path = os.path.join(output_dir, png_base_name.replace('.npy', '.png'))
    plt.savefig(png_output_path)
    print(f"Saved entropy plot to {png_output_path}")
    
    # Generate .tex file
    tex_base_name = png_base_name.replace('.npy', '.tex')
    tex_output_path = os.path.join(output_dir, tex_base_name)
    tex_filename_comment = os.path.splitext(os.path.basename(args.filename))[0].upper()
    tex_content = f"""% {tex_filename_comment}

\\subsection{{Average Entropy Prediction of {args.network} model trained on {args.bins} bin {args.balanced} Dataset}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=1\\linewidth]{{Figures/Results/{png_base_name.replace('.npy', '.png')}}}
    \\caption{{Average Entropy for Correct and Incorrect Steering Angles Predictions for the {args.bins}-Bin {args.network} Model on a {args.balanced} Dataset, with Error Bars Indicating Standard Deviation.}}
    \\label{{fig:{args.bins}_bins_{args.network}_entropy_{args.balanced}}}
\\end{{figure}}
"""
    with open(tex_output_path, 'w') as tex_file:
        tex_file.write(tex_content)
    print(f"Saved LaTeX fragment to {tex_output_path}")  
    plt.show()
    plt.close('all')  # Close all figures to free memory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot entropy distributions for correct and incorrect steering angle predictions.')
    parser.add_argument('--filename', type=str, 
                        default='/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_cnn_softmax_dist_unbalanced.npy',
                        help='Path to the softmax outputs with distances .npy file')
    parser.add_argument('--bins', type=int, default=3, choices=[3, 5, 15],
                        help='Number of steering bins (3, 5, or 15; default: 3)')
    parser.add_argument('--network', type=str, default='cnn',
                        help='Network architecture used (default: cnn)')
    parser.add_argument('--balanced', type=str, default='unbalanced', choices=['balanced', 'unbalanced'],
                        help='Whether the dataset is balanced or unbalanced (default: unbalanced)')
    
    args = parser.parse_args()
    
    try:
        plot_entropy_charts(args)
    except Exception as e:
        print(f"An error occurred: {e}")


"""
# Example usage:
# CNN 3 bins, unbalanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_cnn_softmax_dist_unbalanced.npy \
    --bins 3 \
    --network cnn \
    --balanced unbalanced
# CNN 5 bins, unbalanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_cnn_softmax_dist_unbalanced.npy \
    --bins 5 \
    --network cnn \
    --balanced unbalanced
# CNN 15 bins, unbalanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_cnn_softmax_dist_unbalanced.npy \
    --bins 15 \
    --network cnn \
    --balanced unbalanced
# CNN 3 bins, balanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_cnn_softmax_dist_balanced.npy \
    --bins 3 \
    --network cnn \
    --balanced balanced
# CNN 5 bins, balanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_softmax_dist_balanced.npy \
    --bins 5 \
    --network cnn \
    --balanced balanced
# CNN 15 bins, balanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_cnn_softmax_dist_balanced.npy \
    --bins 15 \
    --network cnn \
    --balanced balanced 
# ViT 3 bins, unbalanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_vit_softmax_dist_unbalanced.npy \
    --bins 3 \
    --network vit \
    --balanced unbalanced
# ViT 5 bins, unbalanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_vit_softmax_dist_unbalanced.npy \
    --bins 5 \
    --network vit \
    --balanced unbalanced
# ViT 15 bins, unbalanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_vit_softmax_dist_unbalanced.npy \
    --bins 15 \
    --network vit \
    --balanced unbalanced
# ViT 3 bins, balanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_softmax_dist_balanced.npy \
    --bins 3 \
    --network vit \
    --balanced balanced
# ViT 5 bins, balanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_softmax_dist_balanced.npy \
    --bins 5 \
    --network vit \
    --balanced balanced
# ViT 15 bins, balanced
python 21-plot-entropy-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_vit_softmax_dist_balanced.npy \
    --bins 15 \
    --network vit \
    --balanced balanced 
"""        

# Alternatively:

"""
#!/bin/bash

# Exit on error
set -e

# Activate the Python environment
source ~/.pyenv/versions/carla-env/bin/activate

# Base directory for datasets
BASE_DIR="/home/daniel/git/neurips-2025/scripts"

# Log file for errors
LOG_FILE="entropy_plot_errors.log"
echo "Starting entropy plot generation at $(date)" > $LOG_FILE

# Array of configurations
CONFIGS=(
    "carla_dataset_640x480_07_3_bins/3_bins_cnn_softmax_dist_unbalanced.npy 3 cnn unbalanced"
    "carla_dataset_640x480_06/5_bins_cnn_softmax_dist_unbalanced.npy 5 cnn unbalanced"
    "carla_dataset_640x480_05/15_bins_cnn_softmax_dist_unbalanced.npy 15 cnn unbalanced"
    "carla_dataset_640x480_07_3_bins_balanced/3_bins_cnn_softmax_dist_balanced.npy 3 cnn balanced"
    "carla_dataset_640x480_06_balanced/5_bins_cnn_softmax_dist_balanced.npy 5 cnn balanced"
    "carla_dataset_640x480_05_balanced/15_bins_cnn_softmax_dist_balanced.npy 15 cnn balanced"
    "carla_dataset_640x480_07_3_bins/3_bins_vit_softmax_dist_unbalanced.npy 3 vit unbalanced"
    "carla_dataset_640x480_06/5_bins_vit_softmax_dist_unbalanced.npy 5 vit unbalanced"
    "carla_dataset_640x480_05/15_bins_vit_softmax_dist_unbalanced.npy 15 vit unbalanced"
    "carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_softmax_dist_balanced.npy 3 vit balanced"
    "carla_dataset_640x480_06_balanced/5_bins_vit_softmax_dist_balanced.npy 5 vit balanced"
    "carla_dataset_640x480_05_balanced/15_bins_vit_softmax_dist_balanced.npy 15 vit balanced"
)

# Loop through configurations
for config in "${CONFIGS[@]}"; do
    # Split config into components
    read filename bins network balanced <<< "$config"
    full_path="${BASE_DIR}/${filename}"
    
    echo "Processing: bins=$bins, network=$network, balanced=$balanced" | tee -a $LOG_FILE
    if [ ! -f "$full_path" ]; then
        echo "Error: File $full_path does not exist" | tee -a $LOG_FILE
        continue
    fi
    
    # Run the Python script
    python 21-plot-entropy-averages.py \
        --filename "$full_path" \
        --bins "$bins" \
        --network "$network" \
        --balanced "$balanced" >> $LOG_FILE 2>&1 || {
        echo "Failed to process $full_path" | tee -a $LOG_FILE
    }
done

echo "Completed entropy plot generation at $(date)" | tee -a $LOG_FILE
"""