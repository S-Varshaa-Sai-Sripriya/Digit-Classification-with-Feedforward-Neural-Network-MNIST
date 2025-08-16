# Digit Classification with Feedforward Neural Network from scratch (MNIST)

Feedforward neural network implemented from scratch using NumPy. Trains on the scikit-learn `digits` MNIST dataset (8x8 images).

---

## Features

**Pure NumPy implementation** — no deep learning libraries  
**Layer-based architecture** (`Dense`, `ReLU`, `Softmax`)  
**Mini-batch gradient descent**  
**Custom training loop** with loss/accuracy tracking  
**Logger & Exception handling**  
**Easily extensible** for more layers or datasets  

---

## Dataset

### MNIST Digit Dataset
The MNIST dataset contains images of handwritten digits (0–9), commonly used for image classification tasks.

- **Number of Instances**: 70,000 (60,000 training, 10,000 testing)  
- **Number of Features**: 784 (28x28 pixels flattened)  
- **Target Variable**: `Digit` (0–9)

**Feature Details**:

- `Pixel1` to `Pixel784` – Grayscale intensity values (0–255) of each pixel in the 28x28 image.  
- `Digit` – The actual digit represented by the image.

This dataset is useful for:

- Training classification models (NN, CNN).  
- Benchmarking image recognition and computer vision algorithms.  

---

## Model Architecture

| Layer        | Details                  |
|--------------|--------------------------|
| Input        | 784 neurons (28×28 pixels) |
| Dense 1      | 128 neurons + ReLU        |
| Dense 2      | 64 neurons + ReLU         |
| Output       | 10 neurons + Softmax      |

---

## Training Results

| Epochs | Train Accuracy | Validation Accuracy |
|--------|---------------|---------------------|
| 30     | **91.6%**     | **90.0%**           |

- **Loss**: decreased from `2.38` → `0.57`  
- Clear convergence without overfitting
