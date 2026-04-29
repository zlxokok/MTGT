
# MTGT

## About MTGT

**MTGT (Multiscale Text Feature-Guided Transformer)** is a deep learning framework for medical image segmentation. It leverages multiscale text features to guide a Transformer architecture, aiming to improve segmentation accuracy on medical imaging datasets.

This repository contains the full implementation of MTGT, including:

- Model code  
- Training scripts  
- Inference scripts  
- Dataset configuration files  

---

# Guides

## Installation

### Software Requirements

- Python 3.x  
- PyTorch (GPU version recommended for training)  
- Other common Python libraries such as:  
  - NumPy  
  - pandas  
  - OpenCV  

> Please check the `import` statements in the code for the complete dependency list.

### Hardware Requirements

- CUDA-capable GPU with sufficient memory recommended  
- Default batch size: **4**

If you encounter **Out of Memory (OOM)** errors, reduce the batch size to **2** by modifying:

- Training script configuration  
- `Config.py`

---

## Obtain MTGT

Clone the repository:

```bash
git clone https://github.com/zlxokok/MTGT.git

Enter the project directory:

cd MTGT

You can now proceed to training or inference.

Usage

MTGT includes:

Training: main_MTGT.py
Inference: infer_MTGT.py
Input Files
Required Data
Medical image datasets
PNG / JPG / other standard image formats
Corresponding segmentation masks
Configuration File
Config.py

Contains:

Dataset paths
Model hyperparameters
Training settings
Data Split Files

For the BUSI dataset:

Train_text.xlsx
Val_text.xlsx

These files define the train/validation split.

MTGT Training

Use main_MTGT.py to train the MTGT model.

python main_MTGT.py
Main Arguments (Configured in Config.py)
Argument	Description
--device	GPU number
--batch_size	Batch size for training
--end_epoch	Number of training epochs
--lr	Learning rate
--checkpoint	Path to save trained model
Main Outputs
Trained model checkpoint
Training logs
Metrics:
Loss
mIoU
MTGT Inference

Use infer_MTGT.py for model inference.

python infer_MTGT.py
Main Arguments (Configured in Config.py)
Argument	Description
--batch_size	Batch size for inference (default: 1)
Main Outputs
Predicted segmentation masks
Evaluation metrics:
mDice
mIoU
Recall
Precision
F1-score
License

This project is publicly shared without a specified license.

Please contact the author for permission if needed.

Contact

For questions or collaborations:

📧 zhaoxuanlong254@gmail.com

Citation

If you use MTGT in your research, please cite:

@article{zhao2026mtgt,
  title={MTGT: Multiscale Text Feature-Guided Transformer in medical image segmentation},
  author={Zhao, L. and Wang, T. and Zhang, X. and others},
  journal={Image and Vision Computing},
  volume={165},
  pages={105846},
  year={2026}
}
