# CNN
# unbalanced 
# 15 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bin_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_cnn_centroids_unbalanced.npy \
    --bins 15 \
    --network cnn

# 5 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bin_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_cnn_centroids_unbalanced.npy \
    --bins 5 \
    --network cnn

# 3 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bin_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_cnn_centroids_unbalanced.npy \
    --bins 3 \
    --network cnn

############
# balanced #
############

# 15 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bin_balanced_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_cnn_centroids_balanced.npy \
    --bins 15 \
    --network cnn \
    --balanced balanced

# 5 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bin_balanced_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_cnn_centroids_balanced.npy \
    --bins 5 \
    --network cnn \
    --balanced balanced

# 3 bins, cnn
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bin_balanced_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_cnn_centroids_balanced.npy \
    --bins 3 \
    --network cnn \
    --balanced balanced

# ViT 
# unbalanced
# 15 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bin_vit_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05/15_bins_vit_centroids_unbalanced.npy \
    --bins 15 \
    --network vit
# 5 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bin_vit_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06/5_bins_vit_centroids_unbalanced.npy \
    --bins 5 \
    --network vit
# 3 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bin_vit_softmax_outputs.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins/3_bins_vit_centroids_unbalanced.npy \
    --bins 3 \
    --network vit

# balanced
# 15 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bin_vit_softmax_outputs_balanced.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_05_balanced/15_bins_vit_centroids_balanced.npy \
    --bins 15 \
    --network vit \
    --balanced balanced
# 5 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bin_vit_softmax_outputs_balanced.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_06_balanced/5_bins_vit_centroids_balanced.npy \
    --bins 5 \
    --network vit \
    --balanced balanced
# 3 bins, vit
python 15-generate-distances-to-centroids.py \
    --filename /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bin_vit_softmax_outputs_balanced.npy \
    --centroids /home/daniel/git/neurips-2025/scripts/carla_dataset_640x480_07_3_bins_balanced/3_bins_vit_centroids_balanced.npy \
    --bins 3 \
    --network vit \
    --balanced balanced