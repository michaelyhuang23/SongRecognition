import numpy as np


def multiply_and_sum(arr: np.ndarray, factor: float) -> float:
    return float(np.sum(arr * factor))
