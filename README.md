# DepthSAM: Camouflaged Object Detection via Depth-Aware Geometric and Semantic Adaptation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official PyTorch implementation of **DepthSAM**, a parameter-efficient framework designed for **Camouflaged Object Detection (COD)**. DepthSAM integrates multiscale geometric depth representations from a frozen **Depth Anything V2** (ViT-Large) encoder with specialized spatial and semantic adapter modules to accurately segment camouflaged targets within complex scenes.

---

## 📌 Architectural Modules

* **Depth Anything V2 (`depth_anything_v2/`):** High-capacity foundation encoder frozen during training to supply dense depth priors without geometric distortion.
* **SMEA (`src/smea.py`):** **Spatial-guided Multi-scale Extraction Adapter** extracts multi-level spatial features from transformer intermediate tokens.
* **GSFM (`src/gsfm.py`):** **Geometric-Semantic Fusion Module** cross-attends and fuses visual semantic tokens with structural depth cues.
* **SFRM (`src/sfrm.py`):** **Spatial Feature Refinement Module** refines fine-grained edge structures and suppresses background clutter.
* **DepthSAM Network (`src/model.py`):** Unified framework combining the frozen backbone, adapter components, and the prediction head.

---

## 📂 Repository Structure

```text
depth_sam_project/
├── dataset/
│   └── COD10K-v3/
│       ├── GT_Object/              # Ground truth binary masks (.png)
│       └── Image/                  # Camouflaged RGB images (.jpg)
├── depth_anything_v2/              # Depth Anything V2 source & dependencies
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # PyTorch Dataset, transforms & DataLoader
│   ├── gsfm.py                     # Geometric-Semantic Fusion Module (GSFM)
│   ├── model.py                    # Core DepthSAM model architecture
│   ├── sfrm.py                     # Spatial Feature Refinement Module (SFRM)
│   └── smea.py                     # Spatial-guided Multi-scale Extraction Adapter (SMEA)
├── test/
│   └── gecko.png                   # Camouflaged sample image for quick testing
├── weights/
│   ├── depth_anything_v2_vitl.pth  # Pretrained Depth Anything V2 backbone
│   ├── depthsam_best.pth           # Checkpoint with lowest validation loss
│   └── depthsam_last.pth           # Latest training checkpoint
├── .gitignore
├── inference.py                    # Raw visual inference pipeline
├── README.md                       # Project documentation
└── train.py                        # Full training pipeline with AMP & gradient accumulation