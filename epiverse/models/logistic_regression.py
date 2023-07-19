import numpy as np
import pandas as pd
from epiverse.models.outcome_model_specification import OutcomeModelSpecification
from epiverse.likelihood.likelihood import Likelihood
from scipy.special import expit


class LogisticRegression(OutcomeModelSpecification):

    def __init__(self, intercept=True):
        self.intercept = intercept
        self.betas = np.array([])

    def fit(self, outcome: np.array, exposure: np.array, **kwargs) -> OutcomeModelSpecification:
        nrows, ncols = exposure.shape

        if self.intercept:
            exposure = np.hstack((np.ones((nrows, 1)), exposure))

        betas = [0] * (ncols + self.intercept)

        def logistic_regression_loglikelihood(parameters, data):
            parameters_as_np = np.array(parameters)

            px = expit(exposure @ parameters)

            ll_contributions = outcome * np.log(px) + \
                (1 - outcome) * np.log(1-px)

            ll_total = np.sum(ll_contributions)
            return ll_total

        logistic_ll = Likelihood(
            loglikelihood_contribution=logistic_regression_loglikelihood
        )

        self.betas = logistic_ll.maximize(initial_values=betas)
        self._is_fit = True
        return self

    def predict(self, observations: np.ndarray | pd.DataFrame):

        if not self._is_fit:
            raise Exception("Logistic regression must first be fit.")
        if isinstance(observations, pd.DataFrame):
            observations = observations.to_numpy()

        if self.intercept:
            nrows, ncols = observations.shape
            observations = np.hstack((np.ones((nrows, 1)), observations))

        return expit(observations @ self.betas)
