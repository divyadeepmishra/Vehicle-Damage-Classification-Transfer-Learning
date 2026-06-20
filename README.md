# Vehicle Damage Classification using Transfer Learning

This project explores transfer learning and fine-tuning for vehicle damage classification using PyTorch and ResNet50. The objective was not only to achieve good accuracy, but also to understand how pretrained computer vision models can be adapted to solve real-world classification problems with limited data.

Instead of training a deep neural network from scratch, a pretrained ResNet50 model was used and fine-tuned on a vehicle damage dataset containing six damage categories.

## Problem Statement

Vehicle damage assessment plays an important role in insurance processing, vehicle inspections, fleet management, and automated reporting systems.

The goal of this project is to classify images into one of the following damage categories:

- Crack
- Scratch
- Tire Flat
- Dent
- Glass Shatter
- Lamp Broken

---

## Dataset

- Total Images: 7,200
- Classes: 6
- Train Split: 80%
- Validation Split: 20%

Images were resized to 224×224 and normalized using ImageNet statistics before being passed to the model.

---

## Model Architecture

### ResNet50 Transfer Learning

The project uses a pretrained ResNet50 model from TorchVision.

Transfer learning strategy:

1. Load pretrained ImageNet weights.
2. Freeze the pretrained feature extraction layers.
3. Replace the original classification head.
4. **Fine-tune Layer4** and the final classifier.
5. Evaluate performance on a validation set.

---

## Results

### Fine-Tuned ResNet50

| Metric | Score |
|----------|----------|
| Accuracy | 96.46% |
| Precision | 96.48% |
| Recall | 96.46% |
| F1 Score | 96.45% |

The largest improvement came from unfreezing and fine-tuning the final residual block (Layer4) rather than training only the classification head.

---

## What I Learned

This project helped me understand:

- Custom PyTorch datasets
- DataLoader pipelines
- Image preprocessing and normalization
- Transfer learning
- Fine-tuning pretrained models
- CrossEntropyLoss
- Adam optimizer
- Accuracy, Precision, Recall and F1 Score
- Confusion Matrix analysis
- Practical computer vision workflows

One of the biggest takeaways was understanding that transfer learning is not about teaching a model vision from scratch. Instead, it is about adapting an already trained visual system to a new task.

---

## Tech Stack

- Python
- PyTorch
- TorchVision
- Pandas
- Scikit-Learn
- PIL
- MPS

---

## Future Improvements

- DenseNet121 comparison
- Data augmentation experiments
- Hyperparameter tuning
- Error analysis and class-wise performance study
- ViT

---

## Project Status

✅ Custom Dataset Pipeline

✅ Transfer Learning

✅ ResNet50 Fine-Tuning

✅ Multi-Class Vehicle Damage Classification

✅ Model Evaluation and Analysis

🚧 DenseNet121 Benchmark (Planned)
