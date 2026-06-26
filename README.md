# 🚗 Vehicle Damage Classification using Transfer Learning

A deep learning project that explores transfer learning for automatic vehicle damage classification using PyTorch.

Instead of training a convolutional neural network from scratch, this project fine-tunes multiple state-of-the-art pretrained models and compares their performance on a real-world multi-class vehicle damage dataset.

The goal of this repository is not only to achieve high accuracy but also to understand how different transfer learning architectures behave on the same dataset.

---

# Project Overview

Vehicle damage assessment is one of the most common computer vision applications in the insurance and automobile industry. Manual inspection is time-consuming and subjective. This project investigates whether pretrained deep learning models can accurately classify different types of vehicle damage.

This repository currently includes experiments with:

- ResNet50
- DenseNet121
- EfficientNet-B0
- Vision Transformer (ViT-B/16)

More architectures will be added as the project progresses.

---

# Dataset

- Total Images: **7,200**
- Classes: **6**

Damage Categories:

- Crack
- Scratch
- Tire Flat
- Dent
- Glass Shatter
- Lamp Broken

Dataset Split:

- Training: **80%**
- Validation: **20%**

---

# Models Implemented

## 1. ResNet50

Transfer Learning

- ImageNet pretrained weights
- Layer4 unfrozen for fine-tuning
- Custom classification head
- PyTorch implementation

### Results

| Metric | Score |
|---------|------:|
| Accuracy | **96.46%** |
| Precision | **96.48%** |
| Recall | **96.46%** |
| F1 Score | **96.45%** |

---

## 2. DenseNet121

Transfer Learning

- ImageNet pretrained weights
- Last Dense Block fine-tuned
- Custom classifier

### Results

| Metric | Score |
|---------|------:|
| Accuracy | **96.32%** |
| Precision | **96.41%** |
| Recall | **96.32%** |
| F1 Score | **96.32%** |

---

## 3. EfficientNet-B0

Transfer Learning

- ImageNet pretrained weights
- Last feature block unfrozen
- Custom classifier

### Results

| Metric | Score |
|---------|------:|
| Accuracy | **75.49%** |
| Precision | **74.02%** |
| Recall | **75.49%** |
| F1 Score | **74.72%** |

Although EfficientNet is known for excellent parameter efficiency, it underperformed on this dataset with the current fine-tuning strategy. This opens opportunities for future experiments involving data augmentation, longer training schedules, and hyperparameter tuning.

---

## 4. Vision Transformer (ViT-B/16)

Transfer Learning using the timm library.

- ImageNet pretrained weights
- Classification head fine-tuned
- Vision Transformer architecture

### Results

| Metric | Score |
|---------|------:|
| Accuracy | **74.10%** |
| Precision | **74.48%** |
| Recall | **74.10%** |
| F1 Score | **73.62%** |

Although Vision Transformers achieve state-of-the-art results on very large datasets, they generally require more data and longer fine-tuning schedules than CNN-based models. On this relatively small dataset, ResNet50 and DenseNet121 significantly outperformed ViT.

---

# Model Comparison

| Model | Accuracy |
|--------|----------:|
| 🥇 ResNet50 | **96.46%** |
| 🥈 DenseNet121 | **96.32%** |
| EfficientNet-B0 | **75.49%** |
| Vision Transformer | **74.10%** |

---

# Tech Stack

- Python
- PyTorch
- Torchvision
- timm
- PIL
- NumPy
- Pandas
- scikit-learn

---

# This project helped me understand much more than transfer learning.

Some of the concepts I explored while building these experiments include:

- PyTorch Dataset and DataLoader
- Transfer Learning
- Fine-Tuning
- Freezing and Unfreezing Layers
- CNN Feature Extraction
- Image Preprocessing
- Model Evaluation
- Precision, Recall and F1 Score
- Confusion Matrix
- Vision Transformers
- Comparative Model Analysis

---
