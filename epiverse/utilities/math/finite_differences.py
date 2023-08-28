from typing import Callable
import numpy as np

ε = 2**-9


def second_central_difference(func: Callable, x0: np.ndarray, epsilon: float = ε) -> np.ndarray:
    rows, cols = np.indices((x0.shape[0], x0.shape[0]))

    def vectorized_central_difference(row: int, column: int):
        h = np.zeros((x0.shape[0]))
        k = np.zeros((x0.shape[0]))

        np.put(h, row, epsilon)
        np.put(k, column, epsilon)

        hk = x0 + h + k
        h += x0
        k += x0

        base_value = func(x0)
        hk_value = func(hk)
        h_value = func(h)
        k_value = func(k)

        return_val = (base_value - h_value - k_value +
                      hk_value) / (epsilon * epsilon)

        return return_val

    return np.vectorize(vectorized_central_difference)(rows, cols)
