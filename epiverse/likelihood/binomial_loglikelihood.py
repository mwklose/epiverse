from epiverse.likelihood.loglikelihood import LogLikelihood
from typing import Callable
from scipy.special import expit
import numpy as np


class BinomialLogLikelihood(LogLikelihood):

    def fit_loglikelihood(self, outcomes: np.ndarray, data: np.ndarray, intercept: bool = True) -> Callable:

        def binomial_loglikelihood(parameters: np.ndarray) -> float:
            if intercept:
                data_with_intercept = np.hstack(
                    (np.Intercept((data.shape[0], 1)), data))
            else:
                data_with_intercept = data

            px = expit(data_with_intercept @ parameters)

            successes = outcomes @ np.log(px)
            failures = (1 - outcomes) @ np.log(1 - px)

            return np.sum(successes + failures)

        return binomial_loglikelihood
