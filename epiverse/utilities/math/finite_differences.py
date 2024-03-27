from typing import Callable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
from icecream import ic


def mv_derivative(func: Callable[[np.ndarray], float], x0: NDArray[np.float64], epsilon: float = 2**-9) -> np.ndarray: 
    """Multivariate Derivative of function, evaluated at vector x0. Implemented as central difference. 

    Args:
        func (Callable): the function to take derivative of, evaluated at x0. 
        x0 (np.ndarray): where to evaluate the derivative at.
        epsilon (float, optional): step size for function evaluations. Defaults to ε.

    Returns:
        float: a float corresponding to the derivative of func at x0. 
    """
    result_array: NDArray = np.zeros(shape=(x0.shape[0], x0.shape[0]))
    x0_reshape: NDArray[np.float64] = np.reshape(x0, (-1, 1))
    
    for i in range(result_array.shape[0]): 
        for j in range(result_array.shape[1]): 
            x0cp: NDArray = x0_reshape.copy()
            
            if i == j: 
                feval: float = func(x0)
                x0cp[i] = x0_reshape[i] + epsilon
                feval_plus: float = func(x0cp)
                x0cp[i] = x0_reshape[i] - epsilon
                feval_minus: float = func(x0cp)
                result_array[i, j] = (feval_plus - 2 * feval + feval_minus) / (epsilon * epsilon)
            elif i < j: 
                x0cp[i] += epsilon
                x0cp[j] += epsilon
                feval_pp = func(x0cp)

                x0cp[j] -= 2 * epsilon
                feval_pm = func(x0cp)

                x0cp[i] -= 2 * epsilon
                feval_mm = func(x0cp)

                x0cp[j] += 2 * epsilon
                feval_mp = func(x0cp)

                result_array[i, j] =  (feval_pp - feval_mp - feval_pm + feval_mm) / (4 * epsilon * epsilon)               
            else: 
                result_array[i, j] = result_array[j, i]
    
    
    return result_array



def second_central_difference(func: Callable, x0: np.ndarray, epsilon: float = 2**-9) -> np.ndarray:
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
