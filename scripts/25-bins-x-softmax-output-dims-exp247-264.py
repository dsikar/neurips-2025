import numpy as np
import pandas as pd

# Experiment dictionary (update with full list if needed)
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
    {"Exp": 264, "Label": "ClSViT5binBalanced_0pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250611_2117.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250611_2117.txt", "Noise_Level": 0, "Bins": 5, "Youtube": "https://youtu.be/d1YI4Eko4JE"},
    {"Exp": 266, "Label": "ClSViT5binBalanced_0pc_pep_noise", "Centroid_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy", "Data_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250612_1032.npy", "Distance_File": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250612_1032.txt", "Noise_Level": 0, "Bins": 5, "Youtube": "https://youtu.be/d1YI4Eko4JE"},
]

results = []
for exp in experiments:
    exp_num = exp["Exp"]
    model = exp["Label"]
    bins = exp["Bins"]
    data_file = exp["Data_File"]
    
    try:
        data = np.load(data_file, allow_pickle=True)
        if len(data) > 0:
            softmax = data[0][0]
            softmax_elements = softmax.shape[-1]
        else:
            raise ValueError("Empty data file")
        results.append({
            "Exp": exp_num,
            "Model": model,
            "Bins": bins,
            "Softmax Elements": softmax_elements
        })
    except Exception as e:
        print(f"Error in Exp {exp_num}: {str(e)}")
        try:
            softmax = data[0][0]
            print(f"Softmax Shape: {softmax.shape}")
            print(f"Softmax Contents: {softmax}")
        except:
            print("Unable to access softmax")
        results.append({
            "Exp": exp_num,
            "Model": model,
            "Bins": bins,
            "Softmax Elements": "Error"
        })

df = pd.DataFrame(results)
print("\nSoftmax Dimension Table:")
print(df.to_string(index=False))

print("\nMismatches:")
for _, row in df.iterrows():
    if row["Softmax Elements"] != "Error" and row["Softmax Elements"] != row["Bins"]:
        print(f"Exp {row['Exp']}: Expected {row['Bins']} bins, got {row['Softmax Elements']} softmax elements")


"""
(trans-env) daniel@simbox ~/git/neurips-2025/scripts (master)$ python 25-bins-x-softmax-output-dims-exp247-264.py 

Softmax Dimension Table:
 Exp                             Model  Bins  Softmax Elements
 237 ClsCNN5binBalanced_10pc_pep_noise     5                 5
 238 ClsCNN5binBalanced_20pc_pep_noise     5                 5
 239 ClsCNN5binBalanced_30pc_pep_noise     5                 5
 240 ClsCNN5binBalanced_40pc_pep_noise     5                 5
 241 ClsCNN5binBalanced_50pc_pep_noise     5                 5
 242 ClsCNN5binBalanced_55pc_pep_noise     5                 5
 243 ClsCNN5binBalanced_60pc_pep_noise     5                 5
 245  ClSViT3binBalanced_1pc_pep_noise     3                 3
 246  ClSViT3binBalanced_2pc_pep_noise     3                 3
 247  ClSViT3binBalanced_3pc_pep_noise     3                 3
 248  ClSViT3binBalanced_4pc_pep_noise     3                 3
 249  ClSViT3binBalanced_5pc_pep_noise     3                 3
 250  ClSViT3binBalanced_7pc_pep_noise     3                 3
 251 ClSViT3binBalanced_10pc_pep_noise     3                 3
 252 ClSViT3binBalanced_20pc_pep_noise     3                 3
 253 ClSViT3binBalanced_30pc_pep_noise     3                 3
 254 ClSViT3binBalanced_40pc_pep_noise     3                 3
 255 ClSViT3binBalanced_50pc_pep_noise     3                 3
 256 ClSViT5binBalanced_10pc_pep_noise     5                 5
 257 ClSViT5binBalanced_20pc_pep_noise     5                 5
 258 ClSViT5binBalanced_30pc_pep_noise     5                 5
 259 ClSViT5binBalanced_40pc_pep_noise     5                 5
 260 ClSViT5binBalanced_50pc_pep_noise     5                 5
 261 ClSViT5binBalanced_60pc_pep_noise     5                 5
 262  ClsCNN5binBalanced_0pc_pep_noise     5                 5
 263  ClSViT3binBalanced_0pc_pep_noise     3                 3
 264  ClSViT5binBalanced_0pc_pep_noise     5                 3

Mismatches:
Exp 264: Expected 5 bins, got 3 softmax elements
"""        