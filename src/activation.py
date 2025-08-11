# src/activations.py
import numpy as np

class ReLU:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        assert self.mask is not None
        grad = dvalues.copy()
        grad[~self.mask] = 0
        return grad
