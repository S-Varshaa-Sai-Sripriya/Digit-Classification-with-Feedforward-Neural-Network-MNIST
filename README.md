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

- **Source**: MNIST (70,000 grayscale 28×28 images of handwritten digits 0–9)
- **Train set**: 60,000 images  
- **Test set**: 10,000 images  

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
