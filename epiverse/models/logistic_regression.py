import numpy as np
import pandas as pd
from epiverse.models.outcome_model_specification import OutcomeModelSpecification
from epiverse.likelihood.binomial_loglikelihood import BinomialLogLikelihood
from scipy.special import expit


class LogisticRegression(OutcomeModelSpecification):

    def __init__(self, intercept=True):
        self.intercept = intercept
        self.betas = np.array([])

    def fit(self, outcome: np.array, exposure: np.array, **kwargs) -> OutcomeModelSpecification:

        bll = BinomialLogLikelihood(
            outcomes=outcome, data=exposure)

        nrows, ncols = bll.data.shape
        betas = [3] * (ncols + self.intercept)

        self.betas = bll.maximize(initial_values=betas)
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
