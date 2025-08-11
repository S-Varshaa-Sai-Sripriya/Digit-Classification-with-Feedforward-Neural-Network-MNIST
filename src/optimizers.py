# src/optimizers.py
import numpy as np
from typing import Optional

class SGD:
    def __init__(self, lr: float = 0.01, momentum: Optional[float] = None) -> None:
        self.lr = lr
        self.momentum = momentum
        # velocity maps will be created lazily keyed by id(layer.weights)
        self.velocity_w = {}
        self.velocity_b = {}

    def step(self, layer, dw: np.ndarray, db: np.ndarray) -> None:
        wid = id(layer.weights)
        bid = id(layer.bias)

        if self.momentum is None:
            layer.weights -= self.lr * dw
            layer.bias -= self.lr * db
        else:
            if wid not in self.velocity_w:
                self.velocity_w[wid] = np.zeros_like(dw)
                self.velocity_b[bid] = np.zeros_like(db)
            self.velocity_w[wid] = self.momentum * self.velocity_w[wid] + self.lr * dw
            self.velocity_b[bid] = self.momentum * self.velocity_b[bid] + self.lr * db
            layer.weights -= self.velocity_w[wid]
            layer.bias -= self.velocity_b[bid]
