from epiverse.models import FunctionModel, LogisticRegression
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from icecream import ic


@dataclass
class PooledLogisticRegression(FunctionModel):
    eqn: str
    time_var: str
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

    def estimate(self, t) -> Callable:
        return lambda v: self.model_dict[t].predict(v)

    def predict(self, t, values, variance=False):
        return self.model_dict[t].predict(values, variance=variance)

    def predict_random(self, t, values, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        predictions = self.model_dict[t].predict(values)
        random_pulls = rng.binomial(1, predictions)
        return random_pulls

    def _fit_procedure(self) -> bool:
        self.model_dict = {}
        for time, group in self.data.groupby(self.time_var):
            self.model_dict[time] = LogisticRegression(self.eqn, group)
        return True
