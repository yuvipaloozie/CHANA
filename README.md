# CHANA: Cell Histology Automated Nuclei Analyzer
### Automated Osteoclast Segmentation and Counting via Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Motivation

In bone pathology research and pharmaceutical development, evaluating osteoclast cell differentiation, morphology, and proliferation is essential for assessing therapeutic efficacy and understanding bone remodeling disorders like osteoporosis. Manual quantification of osteoclasts from micro-photography is constrained by significant operational limitations:

* **Throughput Bottlenecks:** Manual counting and boundary tracing of multi-nucleated cells require substantial time and effort from specialized domain experts, limiting the scale of preclinical trials.
* **Inter-Observer Variability:** Human classification introduces subjective bias, reducing statistical reproducibility across distinct experimental batches and different labs.
* **Complex Morphologies:** Osteoclasts vary significantly in size, shape, and nuclear count, making automated classical thresholding techniques highly unreliable.

**CHANA** addresses these bottlenecks by deploying a fine-tuned Deep Learning framework utilizing a U-Net convolutional neural network topology to execute pixel-perfect semantic segmentation and automated cell counting. This system transforms unstructured biomedical image data into objective, reproducible, and tabular quantitative analytics.

---

## Directory Structure

This repository is currently transitioning from exploratory cloud-hosted notebooks to a production-grade Python package, matching modular software engineering lifecycles.

```text
├── data/
│   ├── raw/                # Original biomedical imagery (not tracked in Git)
│   └── processed/          # Normalized, resized images and binary masks
├── notebooks/              # Google Colab training and exploration artifacts
│   ├── Model Cross Evaluation.ipynb
│   ├── CHANA Inference Notebook.ipynb
├── src/                    # Production core modules (Migration Phase)
│   ├── __init__.py
│   ├── data_loader.py      # Batch generation, tensor scaling, and transforms
│   ├── model.py            # U-Net network topology configuration
│   ├── train.py            # Custom optimization loops and checkpoint tracking
│   └── inference.py        # Scoring pipeline for unindexed microscopy frames
├── config.yaml             # Centralized hyperparameter and directory registry
├── requirements.txt        # Python dependency manifest
└── README.md


