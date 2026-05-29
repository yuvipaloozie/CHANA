```markdown
# CHANA: Deep Learning & Domain Adaptation Pipeline for Automated Quantification of Multinucleated TRAP-Stained Osteoclasts
### Pixel-Level Semantic Segmentation, Curriculum Training, and Instance Separation of Clustered Cells in Digital Pathology

<div align="center">

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Digital_Pathology-red?style=for-the-badge)

</div>

---

## Project Motivation & Scientific Goal

In translational bone pharmacology and preclinical screening for degenerative skeletal disorders (e.g., osteoporosis), evaluating the differentiation and proliferation rate of osteoclasts is essential for validating the efficacy of anti-resorptive therapeutics. Active mature osteoclasts are characterized as large, multi-nucleated cells containing three or more nuclei, traditionally identified via Tartrate-Resistant Acid Phosphatase (TRAP) staining. 

Evaluating these physiological biomarkers under bright-field microscopy faces major operational and technical bottlenecks:

* **High-Latency Experiment Loops:** Manually annotating complex, irregular, and variable cellular boundaries and manually counting thousands of individual instances under a microscope is a high-latency feedback loop that slows down preclinical discovery pipelines.
* **Inter-Observer and Inter-Batch Variance:** Subjective boundary interpretation by different researchers or across separate experiment runs introduces human bias, undermining global statistical reproducibility.
* **Morphological Aggregates:** Active osteoclasts naturally grow in highly dense, overlapping clusters. Classical pixel-intensity thresholding and traditional morphometric software systematically group these clumped structures into monolithic "blobs", causing severe undercounting and morphological estimation errors.

**CHANA** (Computer-assisted Histological Analysis Network for Analytics) addresses these challenges as an end-to-end deep learning framework integrated with an automated domain adaptation pipeline. Developed in collaboration with the Department of Pharmaceutical Sciences at the Philadelphia College of Osteopathic Medicine (PCOM), CHANA maps raw bright-field microscopy images to pixel-level semantic probability maps, separates overlapping boundaries via an inverse-topological distance watershed segmenter, and outputs structured, tabular cell-level metrics.

---

## Mixed-Domain Dataset & Generative Expansion

To train robust, high-capacity deep convolutional networks capable of generalizing across variations in stain intensity, light anomalies, and micro-photography artifacts, CHANA utilizes a mixed-domain dataset:

* **Real Baseline Cohort:** 1,872 bright-field microscopy images containing over 18,000 individual, expert-annotated TRAP-stained osteoclasts.
* **Generative Expansion Matrix:** To bridge the domain gap and enrich boundary representation, a fine-tuned **Stable Diffusion** generative model was engineered to synthesize 3,000 highly realistic cell images paired with pixel-precise binary structural masks.

---

## Directory Structure

This repository provides the production-grade implementation of the CHANA model, translating cloud-trained weights into a modular, deployable command-line tool and interactive web application.

```text
├── .github/workflows/      # Automated CI pipelines for code linting and testing
├── data/
│   ├── raw/                # Unstructured high-resolution .tif microscopy frames (git-ignored)
│   └── processed/          # Preprocessed and standardized 512x512 matrix image-mask pairs
├── notebooks/              # GPU cloud development and model cross-evaluation scripts
│   ├── 01_chana_inference_pipeline.ipynb
│   └── 02_model_cross_evaluation.ipynb
├── src/                    # Standardized production application package
│   ├── __init__.py
│   ├── data_loader.py      # Mixed-domain data pipelines, streaming, and augmentations
│   ├── architectures.py    # Structural definitions (U-Net++, TransUNet, standard U-Net)
│   ├── post_process.py     # Inverse Watershed and EDT instance slicing routines
│   └── evaluation.py       # Centroid-level Hungarian Matching and HD95 validation metrics
├── streamlit_app/          # Production deployment layer
│   └── dashboard.py        # Streamlit interface for full-slide drag-and-drop scoring
├── config.yaml             # Centralized hyperparameter and tiling configurations
├── requirements.txt        # Exact Python package dependencies
└── README.md
```

---

## Core Techniques & Methodology

### 1. 4-Stage Curriculum Learning Paradigm
To incorporate synthetic generative images without introducing training instability, optimization progresses through a structured four-stage curriculum training loop:

```
[ Stage 1: Warmup ] ➔ [ Stage 2: Mixed Training ] ➔ [ Stage 3: Fine Edge Lock ] ➔ [ Stage 4: Hard Mining ]
  Pre-train features     Co-train on mixed real +     Fine-tune exclusively on      Target overlapping,
  on 3,000 synthetic     synthetic image tensors      expert-labeled real           irregular, and faint
  images exclusively.    using heavy augmentations.   microscopy frames.            cell clump boundaries.
```

### 2. Deep Architectural Arena
Three high-capacity semantic segmenters were constructed using a `tf.keras` foundation to identify the optimal framework for modeling long-range spatial context and fine boundary tracking:
* **U-Net++ (Winning Model):** Built with a **ResNet50** backbone, this architecture features nested, dense skip pathways that bridge the semantic gap between encoder and decoder feature maps, capturing multi-scale boundary features.
* **TransUNet:** Blends an **EfficientNetB0** spatial feature extractor with a bottleneck **Vision Transformer (ViT)** block (comprising Multi-Head Self-Attention layers with 6 heads) to retain long-range global relationships.
* **U-Net Baseline:** Configured with a deep **DenseNet121** backbone to establish a rigid spatial benchmark.

### 3. Deep Supervision Joint Optimization
To prevent gradient vanishing across the nested intermediate nodes of the winning U-Net++ architecture, a joint loss function enforces structural alignment. The framework combines pixel-level Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$) with regional structural overlap tracking Dice Loss ($\mathcal{L}_{\text{Dice}}$), distributed dynamically across $M$ operational deep-supervision heads:

$$\mathcal{L}_{\text{Global}} = \sum_{m=1}^{M} w_m \left[ \alpha \mathcal{L}_{\text{BCE}}(y, \hat{y}_m) + \beta \mathcal{L}_{\text{Dice}}(y, \hat{y}_m) \right]$$

---

## Empirical Evaluation & Statistical Benchmarks

Model performance was rigorously cross-evaluated on an independent, unseen holdout vault set. The **U-Net++ architecture trained under the full 4-stage curriculum** demonstrated superior structural and counting performance across all cellular sizes.

### 1. Holdout Set Metrics Summary Table
| Model Architecture & Paradigm | Mean Pixel IoU ↑ | Boundary Error (HD95) ↓ | Count Error (MAE) ↓ | Count Correlation ($R^2$) ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **U-Net++ (Full Curriculum)** 🏆 | **0.669** | **65.9 px** | **1.90 cells** | **0.830** |
| U-Net++ (Baseline) | 0.644 | 72.0 px | 2.12 cells | 0.790 |
| U-Net (Curriculum) | 0.634 | 79.5 px | 2.23 cells | 0.753 |
| U-Net (Baseline) | 0.607 | 86.6 px | 2.49 cells | 0.687 |
| TransUNet (Curriculum) | 0.623 | 78.3 px | 2.26 cells | 0.781 |
| TransUNet (Baseline) | 0.603 | 86.3 px | 2.32 cells | 0.741 |

### 2. Statistical Significance Testing
To mathematically confirm that the performance gains from the curriculum domain adaptation strategy were statistically significant, a non-parametric **Wilcoxon Signed-Rank Test** was performed on the holdout error vectors:
* **U-Net++ Metric Significance:** $p\text{-value} = 1.8969 \times 10^{-13}$ (Extremely Significant)
* **U-Net Metric Significance:** $p\text{-value} = 1.8955 \times 10^{-12}$ (Highly Significant)
* **TransUNet Metric Significance:** $p\text{-value} = 1.4416 \times 10^{-8}$ (Highly Significant)

---

## Full-Slide Tiling & Instance Segmentation Pipeline

```
[ Raw Slide (60+ MP) ] ➔ [ Padding & 512x512 Tiling ] ➔ [ Neural Network Engine ]
                                                                   │
[ Tabular Counts Matrix ] ⟇ [ Area/Shape Filtering ] ⟇ [ Inverse Watershed Slicing ] ⟇ [ EDT Mapping ]
```

To enable inference on large-scale microscopy files (e.g., raw entries exceeding **60.3 Megapixels** at $8056 \times 7480$ resolutions), the production core runs an automated multi-stage instance segmentation post-processing loop:

1. **Sliding Window Tiling Block:** Sections large images into padded $512 \times 512$ pixel matrices, executing parallel GPU processing before stitching them back into a continuous global probability space.
2. **Exact Euclidean Distance Transform (EDT):** Transforms the binary semantic prediction mask into a topographical elevation map by calculating the exact distance from every foreground pixel to its nearest background boundary:
   $$D(x) = \min_{y \in \text{Background}} \|x - y\|_2$$
3. **Marker-Controlled Watershed Slicing:** Locates local peak intensity maxima within the distance map to seed object markers, running an inverse topological watershed algorithm to split touching cell aggregates into distinct instance IDs.
4. **Morphological Filtering Constraints:** Evaluates connected components using `skimage.measure.regionprops`, discards small noise artifacts below an area threshold of 50 pixels, and extracts definitive metrics including total instance counts, cell surface area, circularity, and eccentricity.

---

## Installation & Environment Set Up

To clone this digital pathology workspace and configure your local environment for inference, execute the following commands:

```bash
# Clone the vision asset repository
git clone [https://github.com/your-username/chana-osteoclast-detection.git](https://github.com/your-username/chana-osteoclast-detection.git)
cd chana-osteoclast-detection

# Initialize isolated python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required library distributions
pip install -r requirements.txt

# Execute localized instance pipeline prediction over a sample microscopy image
python src/post_process.py --input data/raw/sample_slide.tif
```

```
