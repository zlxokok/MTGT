# MTGT

**MTGT (Multiscale Text Feature-Guided Transformer)** is a deep learning framework for medical image segmentation. It leverages multiscale text features to guide a Transformer architecture, aiming to improve segmentation accuracy on medical imaging datasets.

This repository contains the full implementation of MTGT, including model code, training and inference scripts, and dataset configuration files.

---

## About

MTGT is a deep learning framework designed for medical image segmentation, utilizing multiscale text features to guide a Transformer-based architecture for improved segmentation accuracy.

---

## Guides

### Installation

#### Software Requirements

1. Python 3.x
2. PyTorch (GPU version recommended for training)
3. Other common Python libraries (e.g., NumPy, pandas, OpenCV, etc.; please check import statements in the code for a full list)

#### Hardware Requirements

1. A CUDA-capable GPU with sufficient memory is recommended.
2. Default batch size: 4. If you run into out-of-memory (OOM) errors, reduce the batch size to 2 by modifying the configuration in the training script or `Config.py`.

### Obtain MTGT

Clone the repository to your local machine using Git:

```bash
git clone https://github.com/zlxokok/MTGT.git



### Citation

If you use MTGT in your research, please cite:

@article{zhao2026mtgt,
  title={MTGT: Multiscale Text Feature-Guided Transformer in medical image segmentation},
  author={Zhao, L. and Wang, T. and Zhang, X. and others},
  journal={Image and Vision Computing},
  volume={165},
  pages={105846},
  year={2026}
}
