# src/utils/io.py
import numpy as np
from typing import List, Tuple

def save_model(path: str, params: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    np.savez_compressed(path, *[arr for pair in params for arr in pair])

def load_model(path: str, model) -> None:
    loaded = np.load(path)
    arrays = [loaded[k] for k in loaded]
    params = [(arrays[i], arrays[i+1]) for i in range(0, len(arrays), 2)]
    model.set_params(params)
