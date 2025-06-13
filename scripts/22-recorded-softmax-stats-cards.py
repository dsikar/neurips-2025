def build_summary(model: str,
                  centroid_path: str,
                  data_path: str,
                  distance_path: str,
                  noise_level: int,
                  bins: int) -> str:
    """
    Returns the formatted, multi-line summary string.
    """
    label = f"{model}_{noise_level}pc_pep_noise"
    return (
        f"Label: {label}\n"
        f"Centroid File: /home/daniel/git/neurips-2025/scripts/{centroid_path}\n"
        f"Data File: /home/daniel/git/neurips-2025/scripts/{data_path}\n"
        f"Distance File: /home/daniel/git/neurips-2025/scripts/{distance_path}\n"
        f"Noise Level: {noise_level}\n"
        f"Bins: {bins}\n"
    )

# 
summary = build_summary(
    model="ClsCNN5binBalanced",
    centroid_path="carla_dataset_640x480_06_balanced/5_bins_centroids_balanced.npy",
    data_path="carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy",
    distance_path="carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.txt",
    noise_level=10,
    bins=5,
)

print(summary)

summary = build_summary(
    model="ClsCNN5binBalanced",
    centroid_path="carla_dataset_640x480_06_balanced/5_bins_centroids_balanced.npy",
    data_path="carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.npy",
    distance_path="carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_10_250609_1903.txt",
    noise_level=10,
    bins=5,
)
print(summary)