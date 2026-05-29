# CHANA: Automated Osteoclast Segmentation & Counting Pipeline

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Digital_Pathology-red?style=for-the-badge)

</div>

---

## Overview

CHANA (Computer-assisted Histological Analysis Network for Analytics) is a deep learning pipeline that automates the segmentation and counting of TRAP-stained osteoclast cells from bright-field microscopy images. 

Developed in collaboration with the Philadelphia College of Osteopathic Medicine (PCOM), this tool replaces tedious manual counting and eliminates human bias. It handles complex, overlapping cell clusters by combining deep semantic segmentation with smart morphological post-processing.

### Key Features
* **Automated Instance Counting:** Accurately separates and counts touching or clumped cells.
* **Hybrid Dataset:** Trained on both real pathology images and high-fidelity synthetic data.
* **Large Image Support:** Uses a sliding-window tiling system to process massive multi-megapixel microscopy slides seamlessly.

---

## The Data

To build a model that handles variations in lighting and stain intensity, we used a combined dataset:
* **Real Images:** 1,872 bright-field images containing >18,000 expert-labeled osteoclast cells.
* **Synthetic Images:** 3,000 realistic images generated using **Stable Diffusion** to expand the training footprint and improve edge detection.

---

## Repository Structure

```text
├── data/
│   ├── raw/                # High-resolution microscopy slides (.tif)
│   └── processed/          # Standardized 512x512 image-mask pairs
├── notebooks/              # Google Colab training & evaluation notebooks
│   ├── 01_chana_inference_pipeline.ipynb
│   └── 02_model_cross_evaluation.ipynb
├── src/                    # Modular production code
│   ├── data_loader.py      # Data generators and image augmentations
│   ├── architectures.py    # Model definitions (U-Net++, TransUNet, U-Net)
│   ├── post_process.py     # Distance Transform and Watershed instance slicing
│   └── evaluation.py       # Validation metrics (IoU, HD95, MAE)
├── streamlit_app/          # UI layer
│   └── dashboard.py        # Web dashboard for drag-and-drop cell counting
├── config.yaml             # Model hyperparameters and tiling parameters
└── requirements.txt        # Python package dependencies
```

---

## How It Works

### 1. Training Workflow (4-Stage Curriculum)
Instead of throwing all data at the model at once, we train it progressively:
* **Stage 1:** Pre-train on the 3,000 synthetic images to learn basic shapes.
* **Stage 2:** Train on a mix of real and synthetic data with aggressive image augmentations.
* **Stage 3:** Fine-tune strictly on real, expert-labeled images to lock in precise cell boundaries.
* **Stage 4:** Optimize specifically on hard examples (weakly stained or highly clustered cells).

### 2. Model Selection
We tested three deep learning architectures optimized using a combined Binary Cross-Entropy and Dice Loss function:
* **U-Net++ (Best Performer):** Uses dense, nested skip connections to accurately map fine cell edges.
* **TransUNet:** Combines an EfficientNet encoder with a Vision Transformer bottleneck for global context.
* **Standard U-Net:** Built with a DenseNet121 backbone as our baseline.

### 3. Post-Processing Pipeline
When working with huge raw slides (up to 60+ Megapixels), the system automatically runs these steps:

```
[ Large Slide ] ➔ [ 512x512 Tiling ] ➔ [ U-Net++ Prediction ] ➔ [ Distance Transform ] ➔ [ Watershed Slicing ] ➔ [ Final Table ]
```

* **Tiling & Stitching:** Breaks huge slides into 512x512 patches for quick GPU processing, then stitches them back together.
* **Distance Transform & Watershed:** Calculates the center peaks of cells to find "seeds," then splits clustered boundaries into individual, uniquely tracked cell instances.
* **Filtering:** Automatically filters out dust and minor background noise artifacts (objects < 50 pixels).

---

## Results & Benchmarks

Evaluated on a completely unseen, real-world holdout vault set, the **U-Net++ model trained with the 4-stage curriculum** delivered the highest accuracy:

| Model Architecture & Strategy | Mean Pixel IoU ↑ | Boundary Error (HD95) ↓ | Count Error (MAE) ↓ | Count Correlation ($R^2$) ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **U-Net++ (4-Stage Curriculum)** 🏆 | **0.669** | **65.9 px** | **1.90 cells** | **0.830** |
| U-Net++ (Standard Baseline) | 0.644 | 72.0 px | 2.12 cells | 0.790 |
| TransUNet (Curriculum) | 0.623 | 78.3 px | 2.26 cells | 0.781 |
| Standard U-Net (Baseline) | 0.607 | 86.6 px | 2.49 cells | 0.687 |

*Note: Statistical significance was validated via a Wilcoxon Signed-Rank Test, confirming that curriculum data adaptation yielded highly reliable improvements (p < 0.001).*

---

## Getting Started

### Installation
Clone the repository and install the dependencies inside a clean virtual environment:

```bash
# Clone the repository
git clone [https://github.com/your-username/chana-osteoclast-detection.git](https://github.com/your-username/chana-osteoclast-detection.git)
cd chana-osteoclast-detection

# Set up and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Running Inference
To run a quick cell segmentation and count check on a raw sample image locally, execute:

```bash
python src/post_process.py --input data/raw/sample_slide.tif
```
