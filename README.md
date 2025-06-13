# 🚗 Self-Driving Neural Networks 🧠

## 🚀 Overview
This repository contains the code for our research on neural networks for autonomous driving. We explore various architectures including CNNs and Vision Transformers (ViT) for steering angle prediction in the CARLA simulator.

## ✅ Key Features
- **Multiple Model Architectures**: NVIDIA-inspired CNNs and Vision Transformers
- **Classification & Regression**: Both approaches for steering angle prediction
- **Robust Evaluation**: Comprehensive metrics including MAE, accuracy, and distance-based measures
- **Uncertainty Analysis**: Softmax output analysis and entropy calculations
- **Noise Resilience Testing**: Performance evaluation under various noise conditions

## 📊 Scripts Overview

### 🔍 Data Processing
- Image loading and preprocessing utilities
- Steering angle extraction and statistics
- Dataset balancing and augmentation

### 🏋️ Training
- Model training from configuration files
- Support for different bin classifications (3, 5, 15 bins)
- Balanced and unbalanced dataset training

### 🧪 Evaluation
- Prediction generation and error computation
- Softmax output analysis and centroid calculations
- Distance metrics (Bhattacharyya, KL divergence, etc.)
- Comprehensive reporting tools (Markdown, LaTeX)

### 🎮 CARLA Integration
- Self-driving implementation in CARLA simulator
- Real-time prediction and control
- Waypoint following and performance tracking

## 📈 Results Visualization
- Softmax averages plotting
- Entropy visualization
- Per-class performance analysis

## 🛠️ Getting Started
Check the scripts directory for numbered examples showing the workflow from data preparation to model evaluation.

## 📝 Citation
If you use this code in your research, please cite our paper:
```
@article{selfdriving2025,
  title={Provisional Title},
  author={Authors},
  journal={Conference/Journal},
  year={2025}
}
```

