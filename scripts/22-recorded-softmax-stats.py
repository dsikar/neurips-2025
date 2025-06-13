# import numpy as np
# data = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy', allow_pickle=True)
# print(data)  # Should show list of (softmax_array, class) tuples
# centroids = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy')


# # Load the data and centroids
# data = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy', allow_pickle=True)
# centroids = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy')

# # Extract the class 0 centroid
# centroid_0 = centroids[0]  # Shape (5,)

# # Initialize list to store distances
# distances = []

# # Iterate through data to find class 0 predictions
# for entry in data:
#     predicted_class = entry[1]  # 6th value is the predicted class index
#     if predicted_class == 0:  # Check for class 0 predictions
#         softmax_output = entry[0][0]  # Extract softmax output (shape (5,))
#         # Compute Euclidean distance to centroid_0
#         distance = np.sqrt(np.sum((softmax_output - centroid_0) ** 2))
#         distances.append(distance)

# # Compute average distance
# if len(distances) > 0:
#     average_distance = np.mean(distances)
#     print(f"Average Euclidean distance to class 0 centroid: {average_distance}")
#     print(f"Number of class 0 predictions: {len(distances)}")
# else:
#     print("No predictions for class 0 found in the data.")

# # average metrics:

# import numpy as np

# # Load the data
# data = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy', allow_pickle=True)

# # Initialize lists to store metrics for class 0 predictions
# bhattacharyya_distances = []
# histogram_intersections = []
# kl_divergences = []

# # Iterate through data to find class 0 predictions
# for entry in data:
#     predicted_class = entry[1]  # 6th value is the predicted class index
#     if predicted_class == 0:  # Check for class 0 predictions
#         bhattacharyya_distances.append(entry[2])  # Bhattacharyya Distance
#         histogram_intersections.append(entry[3])  # Histogram Intersection
#         kl_divergences.append(entry[4])  # KL Divergence

# # Compute average metrics
# if len(bhattacharyya_distances) > 0:
#     avg_bhattacharyya = np.mean(bhattacharyya_distances)
#     avg_histogram = np.mean(histogram_intersections)
#     avg_kl_divergence = np.mean(kl_divergences)
    
#     print(f"Number of class 0 predictions: {len(bhattacharyya_distances)}")
#     print(f"Average Bhattacharyya Distance: {avg_bhattacharyya}")
#     print(f"Average Histogram Intersection: {avg_histogram}")
#     print(f"Average KL Divergence: {avg_kl_divergence}")
# else:
#     print("No predictions for class 0 found in the data.")    


# # Overall

# import numpy as np

# # Load the data and centroids
# data = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy', allow_pickle=True)
# centroids = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy')

# # Process each class (1 to 4)
# for class_index in range(1, 5):
#     # Initialize lists for distances and metrics
#     euclidean_distances = []
#     bhattacharyya_distances = []
#     histogram_intersections = []
#     kl_divergences = []
    
#     # Filter predictions for the current class
#     for entry in data:
#         predicted_class = entry[1]  # Predicted class index
#         if predicted_class == class_index:
#             # Compute Euclidean distance to centroid
#             softmax_output = entry[0][0]  # Softmax output (shape (5,))
#             centroid = centroids[class_index]  # Centroid for current class
#             distance = np.sqrt(np.sum((softmax_output - centroid) ** 2))
#             euclidean_distances.append(distance)
            
#             # Collect metrics
#             bhattacharyya_distances.append(entry[2])  # Bhattacharyya Distance
#             histogram_intersections.append(entry[3])  # Histogram Intersection
#             kl_divergences.append(entry[4])  # KL Divergence
    
#     # Compute averages
#     if len(euclidean_distances) > 0:
#         avg_euclidean = np.mean(euclidean_distances)
#         avg_bhattacharyya = np.mean(bhattacharyya_distances)
#         avg_histogram = np.mean(histogram_intersections)
#         avg_kl_divergence = np.mean(kl_divergences)
        
#         print(f"\nResults for Class {class_index}:")
#         print(f"Number of predictions: {len(euclidean_distances)}")
#         print(f"Average Euclidean distance to centroid: {avg_euclidean}")
#         print(f"Average Bhattacharyya Distance: {avg_bhattacharyya}")
#         print(f"Average Histogram Intersection: {avg_histogram}")
#         print(f"Average KL Divergence: {avg_kl_divergence}")
#     else:
#         print(f"\nNo predictions for Class {class_index} found in the data.")    

# # Single row.     

# import numpy as np

# # Load the data and centroids
# data = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy', allow_pickle=True)
# centroids = np.load('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy')

# # Initialize accumulators
# total_predictions = 0
# sum_euclidean = 0.0
# sum_bhattacharyya = 0.0
# sum_histogram = 0.0
# sum_kl = 0.0

# # Process each prediction
# for entry in data:
#     predicted_class = entry[1]  # Predicted class index
#     softmax_output = entry[0][0]  # Softmax output (shape (5,))
#     centroid = centroids[predicted_class]  # Corresponding centroid
    
#     # Compute Euclidean distance
#     euclidean_distance = np.sqrt(np.sum((softmax_output - centroid) ** 2))
    
#     # Accumulate values
#     sum_euclidean += euclidean_distance
#     sum_bhattacharyya += entry[2]  # Bhattacharyya Distance
#     sum_histogram += entry[3]  # Histogram Intersection
#     sum_kl += entry[4]  # KL Divergence
#     total_predictions += 1

# # Compute averages
# avg_euclidean = sum_euclidean / total_predictions if total_predictions > 0 else 0.0
# avg_bhattacharyya = sum_bhattacharyya / total_predictions if total_predictions > 0 else 0.0
# avg_histogram = sum_histogram / total_predictions if total_predictions > 0 else 0.0
# avg_kl = sum_kl / total_predictions if total_predictions > 0 else 0.0

# # Print results to four decimal places
# print(f"Total Number of Predictions: {total_predictions}")
# print(f"Average Euclidean Distance: {avg_euclidean:.4f}")
# print(f"Average Bhattacharyya Distance: {avg_bhattacharyya:.4f}")
# print(f"Average Histogram Intersection: {avg_histogram:.4f}")
# print(f"Average KL Divergence: {avg_kl:.4f}")   

# # Average distance
# import numpy as np
# perpendicular_distances = np.loadtxt('/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.txt')
# avg_perpendicular = np.mean(perpendicular_distances)
# print(f"Average Perpendicular Distance: {avg_perpendicular:.4f}")

import numpy as np
import pandas as pd

# Experiment data (manually copied from your list for simplicity)
experiments = [
    {"Exp": 237, "Label": "ClsCNN5binBalanced_10pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.txt", "Noise_Level": 10, "Bins": 5, "Youtube": "https://youtu.be/3Zsny4NM_NQ"},
    {"Exp": 238, "Label": "ClsCNN5binBalanced_20pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_20_250609_1940.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_20_250609_1940.txt", "Noise_Level": 20, "Bins": 5, "Youtube": "https://youtu.be/RaCVAlwBQlQ"},
    {"Exp": 239, "Label": "ClsCNN5binBalanced_30pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_30_250609_2008.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_30_250609_2008.txt", "Noise_Level": 30, "Bins": 5, "Youtube": "https://youtu.be/CzJlbYX0CnQ"},
    {"Exp": 240, "Label": "ClsCNN5binBalanced_40pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_40_250609_2056.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_40_250609_2056.txt", "Noise_Level": 40, "Bins": 5, "Youtube": "https://youtu.be/FVlpiNw26J8"},
    {"Exp": 241, "Label": "ClsCNN5binBalanced_50pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_50_250609_2110.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_50_250609_2110.txt", "Noise_Level": 50, "Bins": 5, "Youtube": "https://youtu.be/O74AcmhYF2Y"},
    {"Exp": 242, "Label": "ClsCNN5binBalanced_55pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_55_250609_2121.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_55_250609_2121.txt", "Noise_Level": 55, "Bins": 5, "Youtube": "https://youtu.be/Ui-xJKEpXRs"},
    {"Exp": 243, "Label": "ClsCNN5binBalanced_60pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_60_250609_2134.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_60_250609_2134.txt", "Noise_Level": 60, "Bins": 5},
    {"Exp": 245, "Label": "ClSViT3binBalanced_1pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_1_250610_1241.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_1_250610_1241.txt", "Noise_Level": 1, "Bins": 3},
    {"Exp": 246, "Label": "ClSViT3binBalanced_2pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_2_250610_1443.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_2_250610_1443.txt", "Noise_Level": 2, "Bins": 3},
    {"Exp": 247, "Label": "ClSViT3binBalanced_3pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_3_250610_1455.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_3_250610_1455.txt", "Noise_Level": 3, "Bins": 3},
    {"Exp": 248, "Label": "ClSViT3binBalanced_4pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_4_250610_1500.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_4_250610_1500.txt", "Noise_Level": 4, "Bins": 3},
    {"Exp": 249, "Label": "ClSViT3binBalanced_5pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_5_250610_1502.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_5_250610_1502.txt", "Noise_Level": 5, "Bins": 3},
    {"Exp": 250, "Label": "ClSViT3binBalanced_7pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_7_250610_1506.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_7_250610_1506.txt", "Noise_Level": 7, "Bins": 3},
    {"Exp": 251, "Label": "ClSViT3binBalanced_10pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_10_250610_1508.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_10_250610_1508.txt", "Noise_Level": 10, "Bins": 3},
    {"Exp": 252, "Label": "ClSViT3binBalanced_20pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_20_250610_1516.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_20_250610_1516.txt", "Noise_Level": 20, "Bins": 3},
    {"Exp": 253, "Label": "ClSViT3binBalanced_30pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_30_250610_1520.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_30_250610_1520.txt", "Noise_Level": 30, "Bins": 3},
    {"Exp": 254, "Label": "ClSViT3binBalanced_40pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_40_250610_1535.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_40_250610_1535.txt", "Noise_Level": 40, "Bins": 3},
    {"Exp": 255, "Label": "ClSViT3binBalanced_50pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_50_250610_1541.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_50_250610_1541.txt", "Noise_Level": 50, "Bins": 3, "Youtube": "https://youtu.be/e17e30eX0Rg"},
    {"Exp": 256, "Label": "ClSViT5binBalanced_10pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_10_250610_1557.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_10_250610_1557.txt", "Noise_Level": 10, "Bins": 5},
    {"Exp": 257, "Label": "ClSViT5binBalanced_20pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_20_250610_1559.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_20_250610_1559.txt", "Noise_Level": 20, "Bins": 5},
    {"Exp": 258, "Label": "ClSViT5binBalanced_30pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_30_250610_1601.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_30_250610_1601.txt", "Noise_Level": 30, "Bins": 5},
    {"Exp": 259, "Label": "ClSViT5binBalanced_40pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_40_250610_1602.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_40_250610_1602.txt", "Noise_Level": 40, "Bins": 5},
    {"Exp": 260, "Label": "ClSViT5binBalanced_50pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_50_250610_1604.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_50_250610_1604.txt", "Noise_Level": 50, "Bins": 5},
    {"Exp": 261, "Label": "ClSViT5binBalanced_60pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_60_250610_1607.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_60_250610_1607.txt", "Noise_Level": 60, "Bins": 5, "Youtube": "https://youtu.be/OyENq7Xe88Q"},
    {"Exp": 262, "Label": "ClsCNN5binBalanced_0pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_0_250611_1936.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_0_250611_1936.txt", "Noise_Level": 0, "Bins": 5, "Youtube": "https://youtu.be/vhbmxwMlZfk"},
    {"Exp": 263, "Label": "ClSViT3binBalanced_0pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_0_250611_2039.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_0_250611_2039.txt", "Noise_Level": 0, "Bins": 3, "Youtube": "https://youtu.be/NvsoVrbx9xA"},
    {"Exp": 266, "Label": "ClSViT5binBalanced_0pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250612_1032.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250612_1032.txt", "Noise_Level": 0, "Bins": 5, "Youtube": "https://youtu.be/d1YI4Eko4JE"},
]

# BROKEN {"Exp": 264, "Label": "ClSViT5binBalanced_0pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250611_2117.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250611_2117.txt", "Noise_Level": 0, "Bins": 5, "Youtube": "https://youtu.be/d1YI4Eko4JE"},
# Function to derive Net from Label
def get_net(label):
    return "CNN" if "CNN" in label else "ViT"

# Initialize results
results = []

# Process each experiment
for exp in experiments:
    try:
        # Load files
        data = np.load(exp["Data_File"], allow_pickle=True)
        centroids = np.load(exp["Centroid_File"])
        distances = np.loadtxt(exp["Distance_File"])
        
        # Compute metrics
        euclidean_distances = []
        bhattacharyya_distances = []
        histogram_intersections = []
        kl_divergences = []
        
        for entry in data:
            predicted_class = entry[1]
            softmax_output = entry[0][0]
            centroid = centroids[predicted_class]
            euclidean_distances.append(np.sqrt(np.sum((softmax_output - centroid) ** 2)))
            bhattacharyya_distances.append(entry[2])
            histogram_intersections.append(entry[3])
            kl_divergences.append(entry[4])
        
        # Store results
        results.append({
            "Exp": exp["Exp"],
            "Net": get_net(exp["Label"]),
            "Bins": exp["Bins"],
            "Noise%": exp["Noise_Level"],
            "DistAvg": np.mean(distances),
            "ED": np.mean(euclidean_distances),
            "BD": np.mean(bhattacharyya_distances),
            "HI": np.mean(histogram_intersections),
            "KL": np.mean(kl_divergences),
            "Frames": len(data)
        })
    except Exception as e:
        print(f"Error processing experiment {exp['Exp']}: {str(e)}")
        results.append({
            "Exp": exp["Exp"],
            "Net": get_net(exp["Label"]),
            "Bins": exp["Bins"],
            "Noise%": exp["Noise_Level"],
            "DistAvg": np.nan,
            "ED": np.nan,
            "BD": np.nan,
            "HI": np.nan,
            "KL": np.nan,
            "Frames": np.nan
        })

# Create DataFrame
df = pd.DataFrame(results)

# Sort by Net, Bins, Noise%
df = df.sort_values(by=["Net", "Bins", "Noise%"])

# Round numerical columns to 4 decimal places
for col in ["DistAvg", "ED", "BD", "HI", "KL"]:
    df[col] = df[col].round(4)

# Save to CSV
df.to_csv("experiment_summary.csv", index=False)

# Print table
print(df.to_string(index=False))
