# src/trainer.py
import numpy as np
from typing import Optional, Tuple
from .model import Sequential
from .losses import SoftmaxCrossEntropy
from .optimizers import SGD
from .metrics import accuracy

class Trainer:
    def __init__(self, model: Sequential, optimizer: SGD, loss=SoftmaxCrossEntropy) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def _forward_full(self, X: np.ndarray) -> np.ndarray:
        # Forward through entire model and return logits (final Dense output)
        out = X
        for layer, act in zip(self.model.layers, self.model.activations):
            out = layer.forward(out)
            out = act.forward(out)
        if len(self.model.layers) > len(self.model.activations):
            out = self.model.layers[-1].forward(out)
        return out  # logits

    def fit(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        n = X.shape[0]
        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

        for ep in range(1, epochs + 1):
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            y_shuf = y_onehot[perm]

            batch_losses = []
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]

                # forward -> logits
                logits = self._forward_full(xb)

                # loss forward (softmax + crossentropy)
                loss_val, probs = self.loss.forward(logits, yb)
                batch_losses.append(loss_val)

                # gradient w.r.t. logits
                grad = self.loss.backward(probs, yb)  # shape (batch, classes)

                # Backprop through final dense (if final layer without activation)
                # Note: we must iterate reversed through (maybe last dense) and activations
                # Build reversed iterable aligning with forward mapping:
                # If layers = [L1, L2] activations = [A1] -> forward: L1->A1->L2 (logits)
                # reversed zip(self.model.layers, self.model.activations) gives pairs (L1,A1) only.
                # We'll manually handle reversed pass:
                # Start: grad is dL/dlogits -> pass through last dense
                # We therefore:
                #  - if len(layers)>len(activations) -> last layer has no activation in front of it
                #  - process last dense first, then reversed pairs.

                # handle last dense if exists (no activation)
                if len(self.model.layers) > len(self.model.activations):
                    last_layer = self.model.layers[-1]
                    grad = last_layer.backward(grad)  # grad now dL/dprev
                    # update that layer
                    dw, db = last_layer.grads()
                    self.optimizer.step(last_layer, dw, db)

                # now reversed through remaining (layer, activation) pairs
                for layer, act in reversed(list(zip(self.model.layers, self.model.activations))):
                    grad = act.backward(grad)
                    grad = layer.backward(grad)
                    dw, db = layer.grads()
                    self.optimizer.step(layer, dw, db)

            avg_loss = float(np.mean(batch_losses))
            history['loss'].append(avg_loss)

            # training metrics
            train_probs = self.predict_proba(X)
            train_acc = accuracy(train_probs, y_onehot)
            history['acc'].append(train_acc)

            if X_val is not None and y_val is not None:
                val_logits = self.predict_proba(X_val)
                val_loss, _ = self.loss.forward(val_logits, y_val)
                val_acc = accuracy(val_logits, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                if verbose:
                    print(f"Epoch {ep}/{epochs} — loss: {avg_loss:.4f} — acc: {train_acc:.4f} — val_acc: {val_acc:.4f}")
            else:
                if verbose:
                    print(f"Epoch {ep}/{epochs} — loss: {avg_loss:.4f} — acc: {train_acc:.4f}")

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._forward_full(X)
        # convert logits to probabilities (softmax)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(idx)), idx] = 1
        return onehot
