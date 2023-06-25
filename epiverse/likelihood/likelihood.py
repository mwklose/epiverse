from typing import Callable, List
import scipy.optimize as optim
import numpy as np


class Likelihood:
    def __init__(self, likelihood_contribution: Callable = None, loglikelihood_contribution: Callable = None, data: np.array = None):
        if not likelihood_contribution and not loglikelihood_contribution:
            raise Exception(
                "Must provide either Likelihood or LogLikelihood function.")

        if likelihood_contribution and loglikelihood_contribution:
            raise Exception(
                "Provide either Likelihood or LogLikelihood function, not both. (Cannot check that both functions yield same value)"
            )

        self.likelihood_contribution = likelihood_contribution
        self.loglikelihood_contribution = loglikelihood_contribution
        self.data = data

    def eval(self, parameters: np.array, data: np.array):
        if not self.likelihood_contribution:
            raise Exception("Likelihood function not implemented.")
        return self.likelihood_contribution(data)

    def logeval(self, parameters: np.array, data: np.array):
        if not self.loglikelihood_contribution:
            raise Exception("Loglikelihood function not implemented.")

        return self.loglikelihood_contribution(parameters, data)

    def full_eval(self, parameters: np.array):
        return -1 * self.eval(parameters, self.data)

    def full_logeval(self, parameters: np.array):
        return -1 * self.logeval(parameters, self.data)

    def maximize(self, initial_values: List = [], *args, **kwargs) -> List:
        if not self.likelihood_contribution:
            ll_func = self.full_logeval
        else:
            ll_func = self.full_eval

        optimization_result = optim.minimize(ll_func,
                                             x0=initial_values)
        if not optimization_result.success:
            raise Exception(
                f"Optimization failed, {optimization_result.message}\nLast iter: {optimization_result.x}")

        return optimization_result.x
