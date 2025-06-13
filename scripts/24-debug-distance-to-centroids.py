import numpy as np

# 262 ok, softmax shape (1, 5), 264 broken, 266 ok (264 fixed)
# File paths from your experiment list
data_files = {
    "Exp 262 (CNN, 5 bins)": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_cnn_balanced_pep_0_250611_1936.npy",
    "Exp 264 (ViT, 5 bins)": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250611_2117.npy",
    "Exp 266 (ViT, 5 bins)": "/home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/self_driving_5_vit_balanced_pep_0_250612_1032.npy"
}

for exp_name, data_file in data_files.items():
    print(f"\nInspecting {exp_name}")
    try:
        # Load data
        data = np.load(data_file, allow_pickle=True)
        
        # Print shape
        print(f"Data Shape: {data.shape}")
        
        # Print first row
        if len(data) > 0:
            first_entry = data[0]
            print(f"First Row: {first_entry}")
            print(f"First Row Type: {type(first_entry)}")
            print(f"Softmax Shape: {first_entry[0].shape}")
            print(f"Predicted Class: {first_entry[1]}")
        else:
            print("Error: Data file is empty")
            
    except Exception as e:
        print(f"Error: {str(e)}")



"""
Output:
Inspecting Exp 262 (CNN, 5 bins)
Data Shape: (19749, 5)
First Row: [array([[0.02010538, 0.18086638, 0.5847841 , 0.057106  , 0.15713818]],
       dtype=float32)                                                  2
 np.float64(-0.0016997959956634846) np.float64(1.0039215942274513)
 np.float64(0.0)]
First Row Type: <class 'numpy.ndarray'>
Softmax Shape: (1, 5)
Predicted Class: 2

Inspecting Exp 264 (ViT, 5 bins)
Data Shape: (8768, 5)
First Row: [array([4.0481004e-04, 2.7064655e-03, 9.9688882e-01], dtype=float32) 2
 np.float64(-0.0016997959956632925) np.float64(1.0039215942274509)
 np.float64(0.0)]
First Row Type: <class 'numpy.ndarray'>
Softmax Shape: (3,)
Predicted Class: 2


"""