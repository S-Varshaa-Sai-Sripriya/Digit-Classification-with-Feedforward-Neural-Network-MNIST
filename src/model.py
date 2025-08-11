# src/model.py
from typing import List
import numpy as np

class Sequential:
    """
    Simple sequential container.
    - layers list: Dense layers
    - activations list: activation objects (for hidden layers). The final layer typically has no activation here,
      because loss handles softmax.
    """

    def __init__(self) -> None:
        self.layers: List = []
        self.activations: List = []

    def add(self, layer) -> None:
        self.layers.append(layer)

    def add_activation(self, activation) -> None:
        self.activations.append(activation)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        # apply pairs (layer, activation); there can be one final layer w/o activation
        for layer, act in zip(self.layers, self.activations):
            out = layer.forward(out)
            out = act.forward(out)
        if len(self.layers) > len(self.activations):
            out = self.layers[-1].forward(out)
        return out

    def params(self):
        return [layer.params() for layer in self.layers]

    def set_params(self, params):
        for layer, (w, b) in zip(self.layers, params):
            layer.weights = w
            layer.bias = b
