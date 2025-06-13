import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the image files
directory = 'carla_dataset_640x480_01'

# List to store steering angles
angles = []

# Iterate through files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        try:
            # Extract the steering angle from the filename
            angle_str = filename.split('_')[-1].replace('.jpg', '')
            angle = float(angle_str)
            angles.append(angle)
        except (IndexError, ValueError) as e:
            print(f"Error processing file {filename}: {e}")

# Convert angles list to numpy array for calculations
angles = np.array(angles)

# 1. Compute statistics
if len(angles) > 0:
    mean_angle = np.mean(angles)
    std_angle = np.std(angles)
    variance_angle = np.var(angles)
    min_angle = np.min(angles)
    max_angle = np.max(angles)

    print(f"Statistics for steering angles:")
    print(f"Mean: {mean_angle:.4f}")
    print(f"Standard Deviation: {std_angle:.4f}")
    print(f"Variance: {variance_angle:.4f}")
    print(f"Minimum: {min_angle:.4f}")
    print(f"Maximum: {max_angle:.4f}")
else:
    print("No valid steering angles found in the directory.")

# 2. Plot a histogram (bar chart) with 15 bins
plt.figure(figsize=(10, 6))
plt.hist(angles, bins=15, edgecolor='black')
plt.title('Distribution of Steering Angles, training dataset carla_dataset_640x480_01')
plt.xlabel('Steering Angle')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()
# save the histogram as an image file
plt.savefig('steering_angle_distribution_carla_dataset_640x480_01.png')
# 3. Save the statistics to a text file 
stats_file = 'steering_angle_stats_carla_dataset_640x480_01.txt'
with open(stats_file, 'w') as f:
    f.write(f"Mean: {mean_angle:.4f}\n")
    f.write(f"Standard Deviation: {std_angle:.4f}\n")
    f.write(f"Variance: {variance_angle:.4f}\n")
    f.write(f"Minimum: {min_angle:.4f}\n")
    f.write(f"Maximum: {max_angle:.4f}\n")