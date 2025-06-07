import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Define steering angle labels for each bin configuration
bin_labels = {
    3: [-0.065, 0.0, 0.065],
    5: [-0.065, -0.015, 0.0, 0.015, 0.065],
    15: [-0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005, 0.0, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065]
}

def plot_class_averages(args):
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
    
    # Separate correct and incorrect predictions
    correct_mask = true_labels == pred_labels
    incorrect_mask = ~correct_mask
    correct_data = data[correct_mask]
    incorrect_data = data[incorrect_mask]
    
    # Create a figure and subplots for each class (2 rows: correct and incorrect predictions)
    fig, axs = plt.subplots(2, num_bins, figsize=(num_bins * 2.5, 6))
    fig.suptitle(f'{num_bins} bin {args.balanced} {args.network.upper()} Training Data - Softmax Average Distributions '
                 f'for Correct and Incorrect Steering Angle Predictions', fontsize=16)
    
    # Plot correct predictions
    for i in range(num_bins):
        # Get the softmax outputs for the current class
        class_correct = correct_data[true_labels[correct_mask] == i, :num_bins]
        if len(class_correct) > 0:
            averages = np.mean(class_correct, axis=0)
            axs[0, i].bar(np.arange(num_bins), averages, color='skyblue')
            axs[0, i].set_yscale('log')
            axs[0, i].set_ylim(bottom=1e-4)
            axs[0, i].set_title(f'Steering {bin_labels[num_bins][i]:.3f} (Correct)', fontsize=10)
            axs[0, i].set_xticks(np.arange(num_bins))
            axs[0, i].set_xticklabels([f'{bin_labels[num_bins][j]:.3f}' for j in range(num_bins)], 
                                    fontsize=8, rotation=45)
            axs[0, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
            axs[0, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            axs[0, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
    
    # Plot incorrect predictions
    for i in range(num_bins):
        # Get the softmax outputs for the current class (based on true label)
        class_incorrect = incorrect_data[true_labels[incorrect_mask] == i, :num_bins]
        if len(class_incorrect) > 0:
            averages = np.mean(class_incorrect, axis=0)
            axs[1, i].bar(np.arange(num_bins), averages, color='lightcoral')
            axs[1, i].set_yscale('log')
            axs[1, i].set_ylim(bottom=1e-4)
            axs[1, i].set_title(f'Steering {bin_labels[num_bins][i]:.3f} (Incorrect)', fontsize=10)
            axs[1, i].set_xticks(np.arange(num_bins))
            axs[1, i].set_xticklabels([f'{bin_labels[num_bins][j]:.3f}' for j in range(num_bins)], 
                                    fontsize=8, rotation=45)
            axs[1, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
            axs[1, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            axs[1, i].text(0.5, 0.5, 'No Data', ha='center', va='center')
    
    # Set x-axis label at the bottom of the figure
    fig.text(0.5, 0.04, 'Steering Angle Index', ha='center', fontsize=12)
    # Set y-axis label on the left side of the figure
    fig.text(0.04, 0.5, 'Average Softmax Value (Logarithmic)', va='center', rotation='vertical', fontsize=12)
    
    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.9)  # Adjust the top spacing for the main title
    
    # Generate output directory and filename for PNG
    base_dir = os.path.dirname(os.path.dirname(args.filename))  # Go up to the parent directory
    output_dir = os.path.join(base_dir, 'softmax_results', 'plots')
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    png_base_name = os.path.basename(args.filename).replace('softmax_dist', 'softmax_dist_plot')
    png_output_path = os.path.join(output_dir, png_base_name.replace('.npy', '.png'))
    plt.savefig(png_output_path)
    print(f"Saved plot to {png_output_path}")
    
    # Generate .tex file
    tex_base_name = os.path.basename(args.filename).replace('softmax_dist', 'softmax_dist_plot').replace('.npy', '.tex')
    tex_output_path = os.path.join(output_dir, tex_base_name)
    # Extract filename minus extension and convert to uppercase for comment
    tex_filename_comment = os.path.splitext(os.path.basename(args.filename))[0].upper()
    tex_content = f"""% {tex_filename_comment}

\\subsection{{Average Softmax Prediction of model trained on {args.bins} bin {args.network} {args.balanced} Dataset}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=1\\linewidth]{{Figures/Results/{png_base_name.replace('.npy', '.png')}}}
    \\caption{{Average Softmax Probabilities for Correctly and Incorrectly Classified Steering Angles in the {args.bins} bin {args.network} {args.balanced} training Dataset.}}
    \\label{{fig:{args.bins}_bins_{args.network}_softmax_dist_{args.balanced}}}
\\end{{figure}}
"""
    with open(tex_output_path, 'w') as tex_file:
        tex_file.write(tex_content)
    print(f"Saved LaTeX fragment to {tex_output_path}")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot softmax average distributions for correct and incorrect steering angle predictions.')
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
        plot_class_averages(args)
    except Exception as e:
        print(f"An error occurred: {e}")


"""
# Example usage:
# CNN 3 bins, unbalanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_cnn_softmax_dist_unbalanced.npy \
    --bins 3 \
    --network cnn \
    --balanced unbalanced
# CNN 5 bins, unbalanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_cnn_softmax_dist_unbalanced.npy \
    --bins 5 \
    --network cnn \
    --balanced unbalanced
# CNN 15 bins, unbalanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_cnn_softmax_dist_unbalanced.npy \
    --bins 15 \
    --network cnn \
    --balanced unbalanced
# CNN 3 bins, balanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_cnn_softmax_dist_balanced.npy \
    --bins 3 \
    --network cnn \
    --balanced balanced
# CNN 5 bins, balanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_softmax_dist_balanced.npy \
    --bins 5 \
    --network cnn \
    --balanced balanced
# CNN 15 bins, balanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_cnn_softmax_dist_balanced.npy \
    --bins 15 \
    --network cnn \
    --balanced balanced 
# ViT 3 bins, unbalanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_vit_softmax_dist_unbalanced.npy \
    --bins 3 \
    --network vit \
    --balanced unbalanced
# ViT 5 bins, unbalanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_vit_softmax_dist_unbalanced.npy \
    --bins 5 \
    --network vit \
    --balanced unbalanced
# ViT 15 bins, unbalanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_vit_softmax_dist_unbalanced.npy \
    --bins 15 \
    --network vit \
    --balanced unbalanced
# ViT 3 bins, balanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_softmax_dist_balanced.npy \
    --bins 3 \
    --network vit \
    --balanced balanced
# ViT 5 bins, balanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_softmax_dist_balanced.npy \
    --bins 5 \
    --network vit \
    --balanced balanced
# ViT 15 bins, balanced
python 20-plot-softmax-averages.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_vit_softmax_dist_balanced.npy \
    --bins 15 \
    --network vit \
    --balanced balanced 
"""