# import numpy as np

# # Sample experiments with high ED
# experiments = [
#     {"Exp": 263, "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_0_250611_2039.npy", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bin_vit_softmax_outputs_balanced.npy", "Bins": 3},
#     {"Exp": 264, "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250611_2117.npy", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bin_vit_softmax_outputs_balanced.npy", "Bins": 5},
# ]

# for exp in experiments:
#     print(f"\nInspecting Experiment {exp['Exp']} (Bins: {exp['Bins']})")
#     try:
#         # Load data and centroids
#         data = np.load(exp["Data_File"], allow_pickle=True)
#         centroids = np.load(exp["Centroid_File"])
        
#         # Check centroid shape and sum
#         print(f"Centroid Shape: {centroids.shape}")
#         centroid_sums = np.sum(centroids, axis=1)
#         print(f"Centroid Sums: {centroid_sums}")
        
#         # Inspect first 5 entries
#         for i, entry in enumerate(data[:5]):
#             softmax = entry[0][0]
#             predicted_class = entry[1]
#             centroid = centroids[predicted_class]
            
#             # Check softmax sum and shape
#             softmax_sum = np.sum(softmax)
#             print(f"Entry {i}: Softmax Shape: {softmax.shape}, Sum: {softmax_sum:.4f}")
#             print(f"  Softmax: {softmax}")
#             print(f"  Predicted Class: {predicted_class}, Centroid: {centroid}")
            
#             # Compute Euclidean distance
#             ed = np.sqrt(np.sum((softmax - centroid) ** 2))
#             print(f"  Euclidean Distance: {ed:.4f}")
            
#             # Check if softmax and centroid are valid
#             if not np.isclose(softmax_sum, 1.0, rtol=1e-5):
#                 print("  WARNING: Softmax does not sum to 1")
#             if softmax.shape[0] != exp["Bins"]:
#                 print("  WARNING: Softmax dimension mismatch")
#             if centroid.shape[0] != exp["Bins"]:
#                 print("  WARNING: Centroid dimension mismatch")
                
#     except Exception as e:
#         print(f"Error processing Exp {exp['Exp']}: {str(e)}")

# import numpy as np

# # Sample experiments with negative BD
# experiments = [
#     {"Exp": 262, "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_0_250611_1936.npy"},
#     {"Exp": 263, "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_0_250611_2039.npy"},
# ]

# for exp in experiments:
#     print(f"\nInspecting Experiment {exp['Exp']}")
#     try:
#         # Load data
#         data = np.load(exp["Data_File"], allow_pickle=True)
        
#         # Collect BD values
#         bd_values = [entry[2] for entry in data]
#         print(f"BD Stats: Mean={np.mean(bd_values):.4f}, Min={np.min(bd_values):.4f}, Max={np.max(bd_values):.4f}")
        
#         # Print first 5 entries
#         for i, entry in enumerate(data[:5]):
#             print(f"Entry {i}: BD={entry[2]:.4f}, HI={entry[3]:.4f}, KL={entry[4]:.4f}")
#             if entry[2] < 0:
#                 print("  WARNING: Negative BD detected")
                
#         # Check if BD is consistently negative
#         negative_count = sum(1 for bd in bd_values if bd < 0)
#         print(f"Negative BD Count: {negative_count}/{len(bd_values)}")
        
#     except Exception as e:
#         print(f"Error processing Exp {exp['Exp']}: {str(e)}")        

# import numpy as np
# data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_0_250611_2039.npy"
# data = np.load(data_file, allow_pickle=True)
# for i, entry in enumerate(data[:5]):
#     print(f"Entry {i}: {entry}")
# print(f"Data Shape: {data.shape}, Type: {type(data)}")

# import numpy as np
# centroid_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bin_vit_softmax_outputs_balanced.npy"
# centroids = np.load(centroid_file)
# print(f"Centroid Shape: {centroids.shape}")
# print(f"Centroid Sums: {np.sum(centroids, axis=1)[:5]}")
# print(f"First 5 Centroids: {centroids[:5]}")

# import numpy as np
# centroid_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_centroids_balanced.npy"
# centroids = np.load(centroid_file)
# print(f"Centroid Shape: {centroids.shape}")
# print(f"Centroid Sums: {np.sum(centroids, axis=1)[:5]}")
# print(f"First 5 Centroids: {centroids[:5]}")

# import numpy as np
# centroid_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy"
# centroids = np.load(centroid_file)
# print(f"Centroid Shape: {centroids.shape}")
# print(f"Centroid Sums: {np.sum(centroids, axis=1)[:5]}")
# print(f"First 5 Centroids: {centroids[:5]}")

# import numpy as np

# data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy"
# centroid_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy"

# print("Inspecting Data File for Experiment 237")
# try:
#     # Load data and centroids
#     data = np.load(data_file, allow_pickle=True)
#     centroids = np.load(centroid_file)
    
#     # Print data file info
#     print(f"Data Type: {type(data)}")
#     print(f"Data Shape: {data.shape}")
#     print(f"Data Length: {len(data)}")
    
#     # Print first 5 entries
#     for i, entry in enumerate(data[:5]):
#         print(f"\nEntry {i}:")
#         print(f"  Full Entry: {entry}")
#         print(f"  Softmax: {entry[0]}, Shape: {entry[0].shape}, Sum: {np.sum(entry[0]):.4f}")
#         print(f"  Predicted Class: {entry[1]}")
#         print(f"  BD: {entry[2]:.4f}, HI: {entry[3]:.4f}, KL: {entry[4]:.4f}")
    
#     # Print centroid info
#     print(f"\nCentroid Shape: {centroids.shape}")
#     print(f"Centroid Sums: {np.sum(centroids, axis=1)}")
#     for i, centroid in enumerate(centroids):
#         print(f"Class {i} Centroid: {centroid}, Sum: {np.sum(centroid):.4f}")
        
# except Exception as e:
#     print(f"Error: {str(e)}")

import numpy as np

data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy"
centroid_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy"

print("Computing ED for Experiment 237")
try:
    # Load data and centroids
    data = np.load(data_file, allow_pickle=True)
    centroids = np.load(centroid_file)
    
    # Initialize per-class tracking
    class_counts = {i: 0 for i in range(5)}
    class_distances = {i: [] for i in range(5)}
    
    # Compute ED for each entry
    for i, entry in enumerate(data):
        softmax = entry[0][0]  # Extract 5D vector from (1, 5) array
        predicted_class = int(entry[1])  # Correct index for predicted class
        
        # Validate softmax
        softmax_sum = np.sum(softmax)
        if not np.isclose(softmax_sum, 1.0, rtol=1e-5):
            print(f"Warning: Softmax sum {softmax_sum:.4f} for entry {i}")
            softmax = softmax / softmax_sum  # Normalize
            
        # Validate shapes
        if softmax.shape[0] != 5 or centroids[predicted_class].shape[0] != 5:
            print(f"Shape mismatch at entry {i}: Softmax={softmax.shape}, Centroid={centroids[predicted_class].shape}")
            continue
            
        # Compute Euclidean Distance
        ed = np.sqrt(np.sum((softmax - centroids[predicted_class]) ** 2))
        class_distances[predicted_class].append(ed)
        class_counts[predicted_class] += 1
    
    # Compute per-class averages
    print("\nPer-Class ED Averages:")
    per_class_ed = []
    total_frames = sum(class_counts.values())
    
    for cls in range(5):
        if class_counts[cls] > 0:
            avg_ed = np.mean(class_distances[cls])
            print(f"Class {cls}: Count={class_counts[cls]}, Avg ED={avg_ed:.4f}")
            per_class_ed.append((class_counts[cls], avg_ed))
        else:
            print(f"Class {cls}: Count=0, Avg ED=N/A")
            per_class_ed.append((0, 0))
    
    # Manual sum (weighted average)
    manual_ed = sum(count * ed for count, ed in per_class_ed if count > 0) / total_frames if total_frames > 0 else 0
    print(f"\nManual Weighted ED: {manual_ed:.4f}")
    
    # Code-based sum (overall mean of all EDs)
    all_distances = [ed for cls in class_distances.values() for ed in cls]
    code_ed = np.mean(all_distances)
    print(f"Code Computed ED: {code_ed:.4f}")
    
    # Compare with reported ED
    reported_ed = 0.1787
    print(f"Reported ED: {reported_ed:.4f}")
    if np.isclose(manual_ed, code_ed, rtol=1e-5) and np.isclose(manual_ed, reported_ed, rtol=1e-5):
        print("Manual, Code, and Reported ED match!")
    else:
        print("Mismatch detected:")
        print(f"Manual vs Code: {np.abs(manual_ed - code_ed):.6f}")
        print(f"Manual vs Reported: {np.abs(manual_ed - reported_ed):.6f}")
        
except Exception as e:
    print(f"Error: {str(e)}")


import numpy as np

data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy"
data = np.load(data_file, allow_pickle=True)

bd = np.mean([entry[2] for entry in data])
hi = np.mean([entry[3] for entry in data])
kl = np.mean([entry[4] for entry in data])

print(f"Computed Averages for Exp 237:")
print(f"BD: {bd:.4f} (Table: 0.0352)")
print(f"HI: {hi:.4f} (Table: 0.8305)")
print(f"KL: {kl:.4f} (Table: 0.1797)")    

import numpy as np

data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy"
data = np.load(data_file, allow_pickle=True)

bd = np.mean([entry[2] for entry in data])
hi = np.mean([entry[3] for entry in data])
kl = np.mean([entry[4] for entry in data])

print(f"Computed Averages for Exp 237:")
print(f"BD: {bd:.4f} (Table: 0.0352)")
print(f"HI: {hi:.4f} (Table: 0.8305)")
print(f"KL: {kl:.4f} (Table: 0.1797)")

import numpy as np
def compute_bd(hist1, hist2):
    bc = 0
    for channel in range(3):  # R, G, B
        h1 = hist1[channel] / np.sum(hist1[channel])  # Normalize
        h2 = hist2[channel] / np.sum(hist2[channel])
        bc += np.sum(np.sqrt(h1 * h2))
    bc /= 3  # Average BC across channels
    return -np.log(max(bc, 1e-10))  # Avoid log(0)

import cv2
import numpy as np
def compute_bd_image(img1, img2, bins=256):
    hists1 = [cv2.calcHist([img1], [c], None, [bins], [0, 256]) for c in range(3)]
    hists2 = [cv2.calcHist([img2], [c], None, [bins], [0, 256]) for c in range(3)]
    bc = 0
    for h1, h2 in zip(hists1, hists2):
        h1 = h1 / np.sum(h1)  # Normalize
        h2 = h2 / np.sum(h2)
        bc += np.sum(np.sqrt(h1 * h2))
    bc /= 3
    return -np.log(max(bc, 1e-10))
# Example: img1 = img2 for 0% noise    

data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy"
data = np.load(data_file, allow_pickle=True)
bd_values = [entry[2] for entry in data]
print(f"Exp 237 BD: Mean={np.mean(bd_values):.4f}, Min={np.min(bd_values):.4f}, Max={np.max(bd_values):.4f}")

print("=====")
import numpy as np

data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_0_250611_2039.npy"
centroid_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy"

data = np.load(data_file, allow_pickle=True)
centroids = np.load(centroid_file)

print("Centroids:\n", centroids)
print("Centroid Sums:", np.sum(centroids, axis=1))

for i, entry in enumerate(data[:5]):
    softmax = entry[0]
    pred_class = int(entry[1])
    print(f"\nEntry {i}:")
    print(f"Softmax: {softmax}, Sum: {np.sum(softmax):.4f}")
    print(f"Predicted Class: {pred_class}")
    print(f"Centroid for Class {pred_class}: {centroids[pred_class]}")
    ed = np.sqrt(np.sum((softmax - centroids[pred_class]) ** 2))
    print(f"ED: {ed:.4f}")