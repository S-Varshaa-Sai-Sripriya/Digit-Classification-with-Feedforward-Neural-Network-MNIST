# src/losses.py
import numpy as np
from typing import Tuple

class SoftmaxCrossEntropy:
    """
    Numerically stable Softmax + Categorical Cross-Entropy.
    forward(logits, y_onehot) -> (loss_scalar, probs)
    backward() -> d_logits (batch, classes) = (probs - y)/batch
    """

    @staticmethod
    def forward(logits: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
        # logits: (batch, classes)
        # y_true: (batch, classes) one-hot

        # Numerically stable softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        # Clip to avoid log(0)
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        batch = logits.shape[0]
        loss = -np.sum(y_true * np.log(probs_clipped)) / batch
        return float(loss), probs

    @staticmethod
    def backward(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # gradient w.r.t. logits
        batch = probs.shape[0]
        return (probs - y_true) / batch
