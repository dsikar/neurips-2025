import numpy as np
import pandas as pd
from distance_metrics import compute_average_ed, get_per_class_ed
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

def generate_stats_table(experiments):
    """
    Generate a table with experiment statistics for path distances and softmax metrics.
    
    Args:
        experiments (list): List of dictionaries with experiment details (Exp, Label, 
                           Centroid_File, Data_File, Distance_File, Noise_Level, Bins).
    
    Returns:
        pandas.DataFrame: Table with columns: ExpID, Model, NumBins, NoisePct, AvgPathDist, 
                          StdPathDist, PathDistCount, AvgED, AvgBD, AvgHI, AvgKL, FrameCount.
    """
    try:
        # Initialize results list
        results = []
        
        for exp in experiments:
            exp_id = exp["Exp"]
            try:
                # Extract experiment details
                model = "CNN" if "CNN" in exp["Label"] else "ViT"
                num_bins = exp["Bins"]
                noise_pct = exp["Noise_Level"]
                
                # Compute distance metrics from Distance_File
                with open(exp["Distance_File"], 'r') as f:
                    distances = [float(line.strip()) for line in f if line.strip()]
                if not distances:
                    raise ValueError("Empty Distance_File")
                distances = np.array(distances)
                avg_path_dist = np.mean(distances)
                std_path_dist = np.std(distances, ddof=1)  # Sample standard deviation
                path_dist_count = len(distances)
                
                # Load Data_File and Centroid_File for softmax metrics
                data = np.load(exp["Data_File"], allow_pickle=True)
                centroids = np.load(exp["Centroid_File"])
                frame_count = len(data)
                
                if centroids.shape[0] != num_bins:
                    raise ValueError(f"Centroids shape {centroids.shape} mismatch with {num_bins} bins")
                
                # Compute softmax metrics
                ed_list, bd_list, hi_list, kl_list = [], [], [], []
                for entry in data:
                    softmax = entry[0]
                    predicted_class = int(entry[1])
                    
                    # Handle CNN (1, n) or ViT (n,) softmax
                    softmax = softmax[0] if softmax.ndim == 2 else softmax
                    if softmax.shape[0] != num_bins:
                        continue
                    
                    # Normalize softmax
                    softmax_sum = np.sum(softmax)
                    if not np.isclose(softmax_sum, 1.0, rtol=1e-5):
                        softmax = softmax / softmax_sum
                    
                    centroid = centroids[predicted_class]
                    
                    # Euclidean Distance
                    ed = np.sqrt(np.sum((softmax - centroid) ** 2))
                    ed_list.append(ed)
                    
                    # Bhattacharyya Distance
                    bd = -np.log(np.sum(np.sqrt(softmax * centroid)))
                    bd_list.append(bd)
                    
                    # Histogram Intersection
                    hi = np.sum(np.minimum(softmax, centroid))
                    hi_list.append(hi)
                    
                    # Kullback-Leibler Divergence
                    kl = np.sum(centroid * np.log10((centroid + 1e-10) / (softmax + 1e-10)))
                    kl_list.append(kl)
                
                if not ed_list:
                    raise ValueError("No valid frames processed")
                
                # Compute averages
                avg_ed = np.mean(ed_list)
                avg_bd = np.mean(bd_list)
                avg_hi = np.mean(hi_list)
                avg_kl = np.mean(kl_list)
                
                # Append results
                results.append({
                    "ExpID": exp_id,
                    "Model": model,
                    "NumBins": num_bins,
                    "NoisePct": noise_pct,
                    "AvgPathDist": avg_path_dist,
                    "StdPathDist": std_path_dist,
                    "PathDistCount": path_dist_count,
                    "AvgED": avg_ed,
                    "AvgBD": avg_bd,
                    "AvgHI": avg_hi,
                    "AvgKL": avg_kl,
                    "FrameCount": frame_count
                })
                
            except Exception as e:
                print(f"Error for Exp {exp_id}: {str(e)}")
                results.append({
                    "ExpID": exp_id,
                    "Model": "N/A",
                    "NumBins": 0,
                    "NoisePct": 0,
                    "AvgPathDist": None,
                    "StdPathDist": None,
                    "PathDistCount": 0,
                    "AvgED": None,
                    "AvgBD": None,
                    "AvgHI": None,
                    "AvgKL": None,
                    "FrameCount": 0
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        return df
    
    except Exception as e:
        print(f"Table generation failed: {str(e)}")
        return pd.DataFrame()

df = generate_stats_table(experiments)
print(df.to_string(index=False))
# Optionally save to CSV
df.to_csv("experiment_stats.csv", index=False)


# # 263
result_df = get_per_class_ed(263, experiments)
# Optionally save to CSV
result_df.to_csv("exp_263_per_class_ed.csv", index=False)
