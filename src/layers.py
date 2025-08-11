# src/layers.py
import numpy as np
from typing import Tuple, Optional

class Dense:
    """
    Fully-connected layer with Xavier init.
    - forward: caches input
    - backward: computes gradients (dW, db) averaged over batch and returns dinput
    - grads(): returns (dW, db) for optimizer to use
    Time: forward O(batch * in * out), backward similar.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights: np.ndarray = np.random.uniform(-limit, limit, (in_features, out_features))
        self.bias: np.ndarray = np.zeros((1, out_features))

        # caches / grads
        self.input: Optional[np.ndarray] = None
        self.dweights: Optional[np.ndarray] = None
        self.dbias: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (batch, in_features)
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        dvalues: (batch, out_features) = dL/dZ for this layer
        Computes:
          dW = X^T @ dvalues / batch
          db = sum(dvalues) / batch
        Returns:
          dX = dvalues @ W^T
        """
        assert self.input is not None
        batch = self.input.shape[0]
        self.dweights = (self.input.T @ dvalues) / batch
        self.dbias = np.sum(dvalues, axis=0, keepdims=True) / batch
        dinput = dvalues @ self.weights.T
        return dinput

    def grads(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.dweights is not None and self.dbias is not None
        return self.dweights, self.dbias

    def params(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.weights, self.bias
