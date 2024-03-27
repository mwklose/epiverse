from epiverse.models import FunctionModel
from dataclasses import dataclass
from typing import Callable, Union
from epiverse.likelihood.binomial_loglikelihood import BinomialLogLikelihood

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from icecream import ic


@dataclass
class LogisticRegression(FunctionModel):
    eqn: str
    data: pd.DataFrame

    def __post_init__(self):
        self.outcome, self.covariates = patsy.dmatrices(
            self.eqn, data=self.data)
        self.outcome_vars = self.outcome.design_info.column_names

        if len(self.outcome_vars) != 1:
            raise Exception(
                f"Only one outcome variable allowed. Was {self.outcome_vars}")

        self.covariate_vars = self.covariates.design_info.column_names

        self._is_fit = self._fit_procedure()

    def estimate(self) -> Callable:
        return lambda v: self.model_results.predict(v)

    def predict(self, values, variance=False):
        if variance:
            return self.model_results.predict(values, which="mean"), self.model_results.get_prediction(exog=values).se
        return self.model_results.predict(values, which="mean")

    def predict_random(self, values, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        predictions = self.model_results.predict(values)
        random_pulls = rng.binomial(1, predictions)
        return random_pulls

    def _fit_procedure(self) -> bool:
        self.model_results = sm.GLM(self.outcome, self.covariates,
                                    family=sm.families.Binomial()).fit()

        self.vcov = self.model_results.cov_params()
        self.coefs = self.model_results.params
        return True


@dataclass
class LogisticRegressionByHand(FunctionModel):
    eqn: str
    data: pd.DataFrame

    def __post_init__(self, intercept=True):
        self.outcome, self.covariates = patsy.dmatrices(
            self.eqn, data=self.data)
        self.outcome_vars = self.outcome.design_info.column_names
        self.covariate_vars = self.covariates.design_info.column_names

        self._is_fit, self.betas = self._fit_procedure()

    def estimate(self) -> Callable:
        pass

    def predict(self, t, values):
        # Can't predict baseline, as know that everyone doesn't have outcome at baseline.
        if t == 0:
            return values[self.outcome_vars]

    def _fit_procedure(self) -> bool:
        bll = BinomialLogLikelihood(
            outcomes=self.outcome, data=self.covariates)

        nrows, ncols = bll.data.shape
        betas = [3] * (ncols + self.intercept)

        betas = bll.maximize(initial_values=betas)
        return True, betas

    def predict(self, observations: Union[np.ndarray, pd.DataFrame]):
        if not self._is_fit:
            raise Exception("Logistic regression must first be fit.")
        if isinstance(observations, pd.DataFrame):
            observations = observations.to_numpy()

        if self.intercept:
            nrows, ncols = observations.shape
            observations = np.hstack((np.Intercept((nrows, 1)), observations))

        return expit(observations @ self.betas)
