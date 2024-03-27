from epiverse.models import FunctionModel
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from icecream import ic


@dataclass
class MultioutcomeRegression(FunctionModel):
    eqn: str
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

        self._is_fit, self.models = self._fit_procedure()

    def estimate(self) -> Callable:
        return lambda v: self.predict(v)

    def predict(self, values, variance=True, verbose=False):
        err = np.full((values.shape[0], self.outcome.shape[1]), np.inf)
        pred = np.zeros((values.shape[0], self.outcome.shape[1]))
        sterr = np.zeros((values.shape[0], self.outcome.shape[1]))
        for _ in range(1000, 0, -1):
            for i, ov in enumerate(self.outcome_vars):
                other_outcomes = np.reshape(
                    pred[:, ~i], (-1, len(self.outcome_vars) - 1))
                covs = np.hstack((values, other_outcomes))
                pred[:, i] = self.models[i].predict(covs)
                prediction = self.models[i].get_prediction(exog=covs)
                sterr[:, i] = prediction.se
            diff = err - pred
            if np.all(diff < 1e-9):
                break
            err = pred.copy()
        if verbose:
            print(
                f"MOR prediction: {_} iterations needed to receive {pred}.")

        if variance:
            return pred, sterr
        return pred

    def predict_random(self, values, rng=None):
        if rng == None:
            rng = np.random.default_rng()

        pred, pred_sterr = self.predict(values, variance=True)
        result = np.zeros(pred.shape)
        for i in range(len(self.family_list)):
            if type(self.family_list[i]) is sm.families.Binomial:
                result[:, i] = rng.binomial(1, pred[:, i])
            elif type(self.family_list[i]) is sm.families.Gaussian:
                result[:, i] = rng.normal(pred[:, i], pred_sterr[:, i])
            else:
                raise Exception(
                    f"Prediction in MultioutcomeRegression for {self.family_list[i]} not implemented yet.")

        return result

    def _fit_procedure(self) -> bool:
        models = []
        self.vcov = {}
        for i, ov in enumerate(self.outcome_vars):
            single_outcome = self.outcome[:, i]
            other_outcomes = np.reshape(
                self.outcome[:, ~i], (-1, len(self.outcome_vars) - 1))

            covariates = np.hstack((other_outcomes, self.covariates))
            model = sm.GLM(single_outcome, covariates,
                           family=self.family_list[i]).fit()

            models.append(model)
            self.vcov[ov] = model.cov_params()

        return True, models
