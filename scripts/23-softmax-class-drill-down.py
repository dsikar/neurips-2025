import numpy as np
import pandas as pd

data_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_3_vit_balanced_pep_0_250611_2039.npy"
centroid_file = "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy"

print("Computing Per-Class ED for Experiment 263")
try:
    # Load data and centroids
    data = np.load(data_file, allow_pickle=True)
    centroids = np.load(centroid_file)
    
    # Validate centroid shape
    if centroids.shape != (3, 3):
        raise ValueError(f"Expected centroid shape (3, 3), got {centroids.shape}")
    centroid_sums = np.sum(centroids, axis=1)
    print(f"Centroid Sums: {centroid_sums}")
    
    # Initialize per-class tracking
    class_counts = {i: 0 for i in range(3)}
    class_distances = {i: [] for i in range(3)}
    
    # Compute ED for each entry
    for i, entry in enumerate(data):
        softmax = entry[0]  # Extract 3D vector from (1, 3) array
        predicted_class = int(entry[1])  # Predicted class
        
        # Validate softmax
        if softmax.shape != (3,):
            print(f"Warning: Invalid softmax shape {softmax.shape} at entry {i}")
            continue
        softmax_sum = np.sum(softmax)
        if not np.isclose(softmax_sum, 1.0, rtol=1e-5):
            print(f"Warning: Softmax sum {softmax_sum:.4f} at entry {i}")
            softmax = softmax / softmax_sum  # Normalize
        
        # Compute Euclidean Distance
        ed = np.sqrt(np.sum((softmax - centroids[predicted_class]) ** 2))
        class_distances[predicted_class].append(ed)
        class_counts[predicted_class] += 1
    
    # Compute per-class averages
    results = []
    total_frames = sum(class_counts.values())
    overall_ed = np.mean([ed for cls in class_distances.values() for ed in cls])
    
    for cls in range(3):
        count = class_counts[cls]
        avg_ed = np.mean(class_distances[cls]) if count > 0 else np.nan
        results.append({
            "Class": cls,
            "Count": count,
            "Avg ED": avg_ed,
            "Percentage": (count / total_frames * 100) if total_frames > 0 else 0
        })
    
    # Create table
    df = pd.DataFrame(results)
    df["Avg ED"] = df["Avg ED"].round(4)
    df["Percentage"] = df["Percentage"].round(2)
    
    print("\nPer-Class ED Table for Experiment 263:")
    print(df.to_string(index=False))
    print(f"\nTotal Frames: {total_frames}")
    print(f"Overall ED: {overall_ed:.4f} (Reported: 1.0014)")
    
    # Compare with reported ED
    if np.isclose(overall_ed, 1.0014, rtol=1e-4):
        print("Overall ED matches reported value!")
    else:
        print(f"Mismatch: Computed ED={overall_ed:.4f}, Reported=1.0014")
        
except Exception as e:
    print(f"Error: {str(e)}")