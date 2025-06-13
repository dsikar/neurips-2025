import os
import numpy as np
import argparse
from pathlib import Path

def extract_steering_angles(input_dir, output_dir, file_prefix):
    """
    Scan a folder for .jpg files, extract steering angles, and save to text and .npy files.
    
    Args:
        input_dir (str): Directory containing .jpg files
        output_dir (str): Directory to save output files
        file_prefix (str): Prefix for output file names
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of .jpg files
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    
    # Sort files alphabetically
    jpg_files.sort()
    
    # Extract steering angles
    steering_angles = []
    for file_name in jpg_files:
        try:
            # Extract the part between last underscore and .jpg
            angle_str = file_name.split('_')[-1].replace('.jpg', '')
            # Convert to float
            angle = float(angle_str)
            steering_angles.append(angle)
        except (ValueError, IndexError):
            print(f"Warning: Could not extract steering angle from {file_name}")
            continue
    
    if not steering_angles:
        print("No valid steering angles found")
        return
    
    # Convert to numpy array
    angles_array = np.array(steering_angles)
    
    # Define output file paths
    txt_path = os.path.join(output_dir, f"{file_prefix}_steering_angles.txt")
    npy_path = os.path.join(output_dir, f"{file_prefix}_steering_angles.npy")
    
    # Save to text file
    with open(txt_path, 'w') as f:
        for angle in steering_angles:
            f.write(f"{angle}\n")
    
    # Save to .npy file
    np.save(npy_path, angles_array)
    
    print(f"Saved {len(steering_angles)} steering angles to:")
    print(f"- Text file: {txt_path}")
    print(f"- Numpy file: {npy_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract steering angles from .jpg files')
    parser.add_argument('--input-dir', required=True, help='Input directory containing .jpg files')
    parser.add_argument('--output-dir', required=True, help='Output directory for saving files')
    parser.add_argument('--file-prefix', required=True, help='Prefix for output file names')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the extraction function
    extract_steering_angles(args.input_dir, args.output_dir, args.file_prefix)

if __name__ == "__main__":
    main()

    """
    # Extract ground truth steering angles from .jpg files in a specified directory.
    # Example usage:
    $ python scripts/11-extract-steering-angles.py \
        --input-dir carla_dataset_640x480_01 \
        --output-dir steering_angles_output \ 
        --file-prefix ground_truth_carla_dataset_640x480_01
   """