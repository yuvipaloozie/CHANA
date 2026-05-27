# CHANA: Cellular Histology Automation via Neural Analytics
### Automated Osteoclast Segmentation, Boundary Validation, and Instance Counting in High-Resolution Bio-Microscopy

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## Project Motivation & Business Problem

In bone pathology research and preclinical pharmaceutical validation, assessing osteoclast cell differentiation, multi-nucleated cluster sizes, and global distribution is critical for evaluating the efficacy of anti-resorptive therapeutics (e.g., treating osteoporosis). However, traditional manual quantification introduces major industrial and scientific bottlenecks:

* **High-Latency Feedback Loop:** Manual tracing of complex cell boundaries and manual object counting create an operational lag that dramatically slows preclinical discovery pipelines.
* **Inter-Observer Bias:** Subjective visual thresholds introduce variance across batch runs and laboratory sites, weakening statistical reproducibility.
* **Aggregated/Touching Morphologies:** Osteoclasts naturally grow in dense clusters. Classical pixel-intensity thresholding tools treat clumped structures as monolithic blobs, rendering automated tracking inaccurate.

**CHANA** handles this as a **Virtual Soft-Sensor Pipeline**. By taking massive, unstructured high-resolution microscopy frames, the framework automatically extracts high-contrast structural features, matches pixels using state-of-the-art Deep Learning models, splits overlapping structures via instance segmenters, and outputs structured, tabular cell data.

---

## Directory Structure

This repository maps the cloud-trained model weights and evaluation configurations into a deployment-ready Python engine.

```text
├── .github/workflows/      # Automated linting and test coverage runner
├── data/
│   ├── raw/                # High-resolution microscopy .tif images (git-ignored)
│   └── processed/          # Standardized 512x512 image-mask matrix pairs
├── notebooks/              # GPU cloud development and evaluation scripts
│   ├── 01_chana_inference_pipeline.ipynb
│   └── 02_model_cross_evaluation.ipynb
├── src/                    # Modularized production application package
│   ├── __init__.py
│   ├── preprocessing.py    # V9 "Nuclear Pop" CLAHE + Morphological pipeline
│   ├── architectures.py    # TransUNet, U-Net++, and U-Net builders
│   ├── evaluation.py       # Object-level Hungarian Matching and HD95 engines
│   └── inference_tiling.py # Sequential tile sliding and stitching engine
├── streamlit_app/          # Interactive dashboard system code
│   └── dashboard.py        # Drop-and-serve visual web container
├── config.yaml             # Registry file for pixel thresholds and image dimensions
├── requirements.txt        # System library manifest
└── README.md
```

---

## Core Techniques & Methodology

### 1. Preprocessing Architecture: V9 "Nuclear Pop" Pipeline
Biomedical stains vary across preparation batches. To decouple the network from global variance and highlight sub-cellular targets (nucleation clusters), a structured color-space preprocessing engine was developed:
1. **LAB Color Isolation:** Raw BGR frames are mapped to the **CIE LAB** space to decouple luminosity ($L$) from chrominance details ($A, B$).
2. **Contrast Equalization:** Contrast Limited Adaptive Histogram Equalization (**CLAHE**) is applied to the $L$-channel with a `clipLimit=2.0` and a `tileGridSize=(8, 8)` to suppress local lighting variances.
3. **Morphological Top-Hat Transform:** A morphological ellipse structuring element ($15 	imes 15$ window size) extracts high-frequency, localized variations to generate an isolated `nuclei_map`:
   $$	ext{Top-Hat}(f) = f - (f \circ b)$$
4. **Weighted Blending:** The equalized background and the isolated nuclear signatures are combined linearly to enrich fine internal features:
   $$	ext{Final}_L = 1.0 	imes L_{	ext{CLAHE}} + 0.8 	imes 	ext{Nuclei}_{	ext{Map}}$$
5. **Tensor Normalization:** The channels are recomposed, converted back to RGB, scaled to $[0, 1]$, and standardized using ImageNet metrics ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

### 2. State-of-the-Art Model Arena
Three high-capacity semantic segmenters were implemented using a `tf.keras` foundation to find the best boundary delineation strategy:

* **U-Net++ (Winning Model):** Built upon a **ResNet50** structural core, this design connects the encoder-decoder maps through nested, dense skip pathways, limiting the semantic gap between tracking boundaries.
* **TransUNet:** Blends a standard spatial **EfficientNetB0** feature extractor with a **Vision Transformer (ViT)** bottleneck layer (comprising Multi-Head Attention blocks, 6 heads, and sequence length dimension mapping) to model absolute global context.
* **U-Net:** Implemented using a deep **DenseNet121** backbone framework as a baseline.

### 3. Curriculum Deep Supervision Loss Function
To guarantee convergence across early layers, the models use a multi-head **Deep Supervision** pattern. Rather than computing loss solely on the final layer, auxiliary segmentation outputs ($	ext{aux}_1, 	ext{aux}_2$) provide gradient feeds throughout the network. The combined target loss aggregates Binary Cross-Entropy ($\mathcal{L}_{	ext{BCE}}$) and structural region matching Dice Loss ($\mathcal{L}_{	ext{Dice}}$) across all $M$ operational network heads:

$$\mathcal{L}_{	ext{Global}} = \sum_{m=1}^{M} w_m \left[ lpha \mathcal{L}_{	ext{BCE}}(y, \hat{y}_m) + eta \mathcal{L}_{	ext{Dice}}(y, \hat{y}_m) 
ight]$$

---

## Rigorous Evaluation Metrics & Statistical Validation

To pass strict pharmaceutical validation benchmarks, the models are evaluated beyond basic pixel-accuracy metrics using a multi-tiered validation approach on a 281-image **Holdout Vault Set**:

### 1. Boundary & Object Matching Constraints
* **95th Percentile Hausdorff Distance (HD95):** Measures structural distance errors between true and estimated edge contours, strictly penalizing localized boundary displacement.
* **Hungarian Object Matching (Linear Assignment):** To capture point-level tracking performance, object centroids are matched using a distance cost matrix. An instance is classified as a True Positive ($	ext{TP}$) if the Euclidean distance error falls within $	au \le 25$ pixels:
   $$\min \sum_{i} \sum_{j} C_{ij} X_{ij}$$

### 2. SOTA Benchmark Summary Table
The benchmark evaluation demonstrates that applying a **Curriculum Learning** weight-initialization strategy provides systematic gains across every model architecture:

| Model Architecture & Paradigm | Mean Pixel IoU ↑ | Boundary Error (HD95) ↓ | Count Error (MAE) ↓ | Object F1-Score ↑ | Count Variance ($R^2$) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **U-Net++ (Curriculum)** 🏆 | **0.669** | **63.9 px** | **1.90 cells** | **0.833** | **0.829** |
| U-Net++ (Baseline) | 0.644 | 72.0 px | 2.12 cells | 0.798 | 0.790 |
| U-Net (Curriculum) | 0.634 | 79.5 px | 2.23 cells | 0.808 | 0.753 |
| U-Net (Baseline) | 0.607 | 86.6 px | 2.49 cells | 0.772 | 0.687 |
| TransUNet (Curriculum) | 0.623 | 78.3 px | 2.26 cells | 0.765 | 0.781 |
| TransUNet (Baseline) | 0.603 | 86.3 px | 2.32 cells | 0.758 | 0.741 |

### 3. Statistical Significance Validation
A non-parametric **Wilcoxon Signed-Rank Test** was computed on the test distribution to confirm that performance gains from curriculum training methods were not random variations.
* **TransUNet:** $p	ext{-value} = 1.4416 	imes 10^{-8}$  (Highly Significant)
* **U-Net:** $p	ext{-value} = 1.8955 	imes 10^{-12}$  (Highly Significant)
* **U-Net++:** $p	ext{-value} = 1.8969 	imes 10^{-13}$  (Highly Significant)

---

## Full-Slide Tiling & Instance Segmentation Deployment

```
[ Raw Slide (60+ MP) ] ➔ [ Padding & 512x512 Tiling ] ➔ [ Neural Network Engine ]
                                                                   │
[ Final Count Table ] ⟇ [ Area Filter & regionprops ] ⟇ [ Distance/Watershed Separation ]
```

To parse ultra-large raw microscopy records (e.g., an 8K image path tracking a **60.3 Megapixel** micro-photograph container at $8056 	imes 7480$ resolution), the production script initializes an end-to-end multi-stage inference package:

1. **Sliding Window Tiling Block:** Dynamically appends zero-value reflection padding borders to match block sizing, sequentially processing 240 distinct sub-patches across the slide tensor in under 32 seconds.
2. **Stitching Loop:** Recomposes patch output tensors back into a unified probability space.
3. **Distance Transform Map:** Appends structural morphological hole-filling, before running an **Exact Euclidean Distance Transform (EDT)** to convert spatial targets into topological maps:
   $$D(x) = \min_{y \in 	ext{Background}} \|x - y\|$$
4. **Watershed Separation:** Locates local maxima peaks (`min_distance=20` separation parameter) to inject distinct marker seeds, running an inverse topological **Watershed Segmentation** algorithm to slice touching boundaries safely into clean instance IDs.
5. **Area Filtering Constraints:** Extracts properties via `skimage.measure.regionprops`, drops noise elements below a threshold of `min_area=50` pixels, and outputs a clean dataset with exact cell counts, sizes, and localization coordinates.

---

## Installation & Environment Set Up

```bash
# Clone the vision asset repository
git clone https://github.com/your-username/chana-histological-automation.git
cd chana-histological-automation

# Activate your project layout environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Core dependencies setup
pip install -r requirements.txt
```

---

## Future Operational Roadmap

1. **Synthetic Sample Generation via Constrained VAEs:** Introduce Variational Autoencoders constrained by structural topological parameters to synthetically generate edge-case cell models, minimizing boundary tracing bottlenecks.
2. **Operator Procedural Flowcharts:** Deconstruct the high-dimension multi-head weight parameters into structured decision-tree diagrams to build physical tracking documentation for laboratory technicians.
