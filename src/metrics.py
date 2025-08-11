# src/metrics.py
import numpy as np

def accuracy(y_pred_probs: np.ndarray, y_true_onehot: np.ndarray) -> float:
    preds = np.argmax(y_pred_probs, axis=1)
    true = np.argmax(y_true_onehot, axis=1)
    return float(np.mean(preds == true))
