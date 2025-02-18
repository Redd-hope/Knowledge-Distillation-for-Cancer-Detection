# **Knowledge Distillation for Cancer Detection**

This project implements **Knowledge Distillation** to train a lightweight **Student Model** for cancer detection using a powerful **Teacher Model** that combines **MViT, ViT, YOLO, and CNN architectures**. The Student Model is optimized for **embedded systems** and **real-time inference** while maintaining high accuracy.

---

## 🚀 Project Overview

### 🔹 Teacher Model
The **Teacher Model** is an ensemble of powerful deep learning models:
- **MViT (Multi-scale Vision Transformer v2)** – Extracts spatiotemporal features from video data.
- **ViT (Vision Transformer)** – Captures global dependencies in image data.
- **YOLOv8** – Performs fast object detection to enhance feature extraction.
- **ResNet-18 (CNN)** – Provides strong feature representations.

All features are fused and passed through a classification head for **binary classification (Cancer/No Cancer).**

### 🔹 Student Model
A lightweight **CNN-based model** is trained using **Knowledge Distillation** to approximate the Teacher Model's knowledge while being optimized for **embedded devices**.

### 🔹 Knowledge Distillation
A **custom distillation loss** function combines:
1. **KL Divergence Loss** – Encourages the student to mimic the teacher's soft outputs.
2. **Cross-Entropy Loss** – Ensures correct classification.

---

## 📂 Project Structure

```plaintext
📦 Knowledge-Distillation-Cancer-Detection
│── 📂 models
│   ├── teacher_model.py  # Defines the Teacher Model
│   ├── student_model.py  # Defines the Student Model
│── 📂 utils
│   ├── distillation_loss.py  # Knowledge Distillation Loss function
│   ├── dataset.py  # data loading
│── 📂 tests
│   ├── test_models.py  # Unit tests for models
│── main.py  # Entry point for training and evaluation
│── requirements.txt  # Dependencies for the project
│── README.md  # Project documentation
│── test.py
│── train.py
│── config.py
