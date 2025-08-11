# src/utils/data.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

def load_digits_data(test_size: float = 0.2, normalize: bool = True, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_digits()
    X = data.data.astype(float)  # (n_samples, 64)
    y = data.target.reshape(-1, 1)
    if normalize:
        X = X / 16.0  # pixels are 0-16

    # OneHotEncoder compatibility across sklearn versions
    try:
        enc = OneHotEncoder(sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(sparse=False)
    y_onehot = enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test
