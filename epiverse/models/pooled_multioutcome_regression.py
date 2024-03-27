from epiverse.models import FunctionModel
from epiverse.models import MultioutcomeRegression
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from icecream import ic


@dataclass
class PooledMultioutcomeRegression(FunctionModel):
    eqn: str
    time_var: str
    data: pd.DataFrame
    family_list: List[sm.families.Family]

    def __post_init__(self):
        self.outcome, self.covariates = patsy.dmatrices(
            self.eqn, data=self.data)

        self.outcome_vars = self.outcome.design_info.column_names

        if len(self.outcome_vars) == 1:
            raise Exception(
                f"One outcome variable provided; use a better method.")

        self.covariate_vars = self.covariates.design_info.column_names

        self._is_fit = self._fit_procedure()

    def estimate(self, t) -> Callable:
        return lambda v: self.model_dict[t].predict(v)

    def predict(self, t, values, variance=True, verbose=False):
        return self.model_dict[t].predict(values, variance=variance, verbose=verbose)

    def predict_random(self, t, values, rng=None):
        return self.model_dict[t].predict_random(values, rng=rng)

    def _fit_procedure(self) -> bool:
        self.model_dict = {}

        for t, group in self.data.groupby(self.time_var):
            self.model_dict[t] = MultioutcomeRegression(
                self.eqn, group, self.family_list)
        return True
