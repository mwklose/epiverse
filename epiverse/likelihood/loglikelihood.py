from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Callable, List
import scipy.optimize as optim
import numpy as np
from epiverse.utilities.math.finite_differences import second_central_difference

# TODO: change this functionality so it has elements of partial evaluation.
# So, outer likelihood class takes in likelihood contribution and data, then can evaluate at individual points or a set of points.


@dataclass
class LogLikelihood(ABC):
    outcomes: np.ndarray
    data: np.ndarray
    loglikelihood_contribution: Callable = None
    loglikelihood_function: Callable = field(init=False)
    n_parameters: int = field(init=False)

    def __post_init__(self):
        if self.data.shape[0] != self.outcomes.shape[0]:
            raise Exception(
                f"Outcome array shape ({outcome_array.shape}) must be compatible with data shape ({self.data.shape})")

        self.loglikelihood_function = self.fit_loglikelihood(
            self.outcomes, self.data)
        self.n_parameters = self.data.shape[1]

    @abstractmethod
    def fit_loglikelihood(self, outcomes: np.ndarray, data: np.ndarray) -> Callable:
        pass

    def __call__(self, parameters) -> float:
        return self.eval(parameters)

    def eval(self, parameters) -> float:
        return self.loglikelihood_function(parameters)

    def negative_loglikelihood(self, parameters) -> float:
        return -1 * self.loglikelihood_function(parameters)

    def variance(self, parameters) -> np.ndarray:
        # Variance is second derivative of loglikelihood function.

        pass

    def maximize(self, initial_values: List = [], *args, **kwargs) -> List:
        nelder_mead_optim = optim.minimize(self.negative_loglikelihood, method="Nelder-Mead",
                                           x0=initial_values)

        bfgs_optim = optim.minimize(self.negative_loglikelihood,
                                    x0=nelder_mead_optim.x)

        if not bfgs_optim.success:
            raise Exception(
                f"Optimization failed, {bfgs_optim.message}\nLast iter: {bfgs_optim.x}")

        return bfgs_optim.x
