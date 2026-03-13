# Deep Learning Course Project Context

## Project Overview

This project is for a **graduate Deep Learning course**. The goal is to build and evaluate deep learning models for **satellite land-use classification** using the **EuroSAT dataset**.

The project will compare **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** on the same dataset and analyze performance across several dimensions.

The experiments will be implemented in **PyTorch** and trained on a **GPU (RTX 5080)**.

---

# Dataset: EuroSAT

## Description

The **EuroSAT dataset** is a benchmark dataset for **land use and land cover classification** derived from satellite imagery captured by Sentinel-2 satellites. ([Hugging Face][1])

Key characteristics:

* **27,000 labeled images**
* **10 land-use classes**
* **Image resolution: 64×64 pixels**
* Derived from **Sentinel-2 satellite imagery**
* Covers images from **34 European countries**

The dataset exists in two versions:

| Version       | Description             |
| ------------- | ----------------------- |
| RGB           | 3-band images (R, G, B) |
| Multispectral | 13 spectral bands       |

For this project we will likely use the **RGB version**.

---

## Classes

The dataset contains the following 10 classes:

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake ([Hugging Face][2])

Each image is a **satellite image patch representing a small geographic region**.

---

# Problem Definition

**Task:**
Perform **multi-class image classification** of satellite images to predict the land-use class.

Input:

```
Satellite image (64×64 RGB)
```

Output:

```
Predicted land use category
```

Example:

```
Input → CNN / ViT → "Residential"
```

---

# Project Goal

Evaluate how different deep learning architectures perform on satellite imagery.

Main research questions:

1. Do **Vision Transformers outperform CNNs** on satellite imagery?
2. How much does **transfer learning improve performance**?
3. How does **model size affect performance and compute cost**?

---

# Experiments

The project will include **three major comparisons**.

---

# Experiment 1 — CNN vs Vision Transformer

### Goal

Compare traditional convolutional models with transformer-based models.

### Models

CNN

* ResNet18
* ResNet50

Transformer

* ViT-Tiny
* ViT-Base

### Metrics

* Accuracy
* Precision / Recall
* Confusion matrix
* Training time

### Expected Insight

CNNs may perform similarly or better due to smaller dataset size.

Transformers often require larger datasets.

---

# Experiment 2 — Transfer Learning vs Training from Scratch

### Goal

Determine how pretraining affects performance.

### Setup

| Model    | Training            |
| -------- | ------------------- |
| ResNet18 | From scratch        |
| ResNet18 | ImageNet pretrained |
| ViT      | Pretrained          |

### Metrics

* Accuracy
* Convergence speed
* Training stability

### Expected Insight

Pretrained models should:

* converge faster
* achieve higher accuracy

---

# Experiment 3 — Model Capacity vs Performance

### Goal

Study scaling effects.

### Models

| Model    | Type              |
| -------- | ----------------- |
| ResNet18 | small CNN         |
| ResNet50 | larger CNN        |
| ViT-Tiny | small transformer |
| ViT-Base | large transformer |

### Metrics

* Accuracy
* Model parameters
* Inference time
* GPU memory usage

### Expected Output

Plot:

```
Accuracy vs Model Size
```

---

# Model Architectures

## CNN Baseline

### ResNet18

Reason:

* standard benchmark
* relatively small
* easy to train
* widely used baseline

Parameters:

```
~11M parameters
```

---

### ResNet50

Reason:

* deeper CNN
* used for scaling experiment

Parameters:

```
~25M parameters
```

---

# Vision Transformers

Vision Transformers split an image into patches and process them as tokens.

Basic pipeline:

```
Image
↓
Patch embedding
↓
Transformer encoder
↓
Classification token
↓
Softmax
```

---

## ViT Model Sizes

| Model     | Layers | Hidden Dim | Params |
| --------- | ------ | ---------- | ------ |
| ViT-Tiny  | 12     | 192        | ~5M    |
| ViT-Small | 12     | 384        | ~22M   |
| ViT-Base  | 12     | 768        | ~86M   |

For this project:

```
ViT-Tiny
ViT-Base
```

---

# Patch Tokenization

Example for EuroSAT images:

```
Image size: 64×64
Patch size: 16
```

Number of tokens:

```
(64 / 16)^2 = 16 tokens
```

Plus classification token.

This makes computation **very efficient compared to typical ViT training**.

---

# Hardware

GPU:

```
RTX 5080
```

Expected training times:

| Model    | Estimated Time |
| -------- | -------------- |
| ResNet18 | ~20 min        |
| ResNet50 | ~40 min        |
| ViT-Tiny | ~30 min        |
| ViT-Base | ~1 hour        |

Total project training time:

```
~4–6 hours
```

---

# Training Configuration

Typical training setup:

```
batch size = 128
epochs = 30–50
optimizer = AdamW
learning rate = 3e-4
loss = CrossEntropy
```

Data augmentation:

* random flips
* rotations
* normalization

---

# Evaluation Metrics

Metrics to report:

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix

Additional analysis:

* training curves
* model parameter count
* inference latency

---

# Expected Figures for Report

Figures to generate:

1. Training vs validation accuracy
2. Confusion matrix
3. Accuracy vs model size
4. Sample predictions
5. Attention maps (for ViT)

---

# Project Pipeline

Overall system:

```
Dataset
 ↓
Data preprocessing
 ↓
Model training
 ↓
Evaluation
 ↓
Analysis
```

---

# Possible Extensions

If time allows:

* visualize transformer attention maps
* evaluate robustness to noise
* try multispectral EuroSAT

---

# Implementation Libraries

Recommended libraries:

```
PyTorch
timm
torchvision
scikit-learn
matplotlib
```

Example model loading:

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True)
```

---

# Summary

This project investigates **satellite land-use classification using deep learning**, comparing:

* CNN architectures
* Vision Transformers
* Transfer learning effects
* Model scaling

Dataset:

```
EuroSAT
```

Models:

```
ResNet18
ResNet50
ViT-Tiny
ViT-Base
```

Experiments:

1. CNN vs Transformer
2. Transfer learning vs scratch
3. Model capacity vs performance

Hardware:

```
RTX 5080
```

---

If you'd like, I can also generate a **second file (`PROJECT_PLAN.md`) that outlines the exact repo structure and Codex prompts to automatically generate most of the code for this project.**

[1]: https://huggingface.co/datasets/GFM-Bench/EuroSAT?utm_source=chatgpt.com "GFM-Bench/EuroSAT · Datasets at Hugging Face"
[2]: https://huggingface.co/datasets/blanchon/EuroSAT_RGB?utm_source=chatgpt.com "blanchon/EuroSAT_RGB · Datasets at Hugging Face"
