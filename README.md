# 🌟 CNN Introduction Project — CIFAR-10 Classification

This project is my first hands-on practice with Machine Learning (ML) and Deep Learning (DL) concepts using TensorFlow and Keras.  
It focuses on building and training a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

---

## 🎯 Objectives
- Understand the basic structure of a **CNN** (convolution, pooling, dense layers, etc.).
- Learn how to **train**, **evaluate**, and **visualize** a model.
- Get familiar with key ML/DL concepts: normalization, loss, accuracy, epochs, overfitting.
- Prepare myself for more advanced AI projects (image recognition, medical imaging, etc.).

---

## 🧠 Dataset
- **CIFAR-10**: 60,000 color images (32×32 pixels) across 10 classes:
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

---

## ⚙️ Model Architecture
- 3 Convolution blocks (Conv2D + BatchNormalization + MaxPooling)
- Flatten + Dense(512) + Dropout
- Output layer with 10 neurons (Softmax activation)

---

## 🚀 Training
- Optimizer: `Adam`
- Loss function: `Sparse Categorical Crossentropy`
- Metric: `Accuracy`
- Trained for **20 epochs**

---

## 📈 Results
After training, the model achieves a good accuracy on the test set and can correctly predict most image classes.
