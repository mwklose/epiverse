import numpy as np
import pandas as pd

from epiverse.models import FunctionModel
from dataclasses import dataclass
from typing import Union, Callable, Tuple


@dataclass
class KaplanMeier(FunctionModel):
    time: Union[np.ndarray, pd.Series]
    delta: Union[np.ndarray, pd.Series]
    weights: Union[np.ndarray, pd.Series]
    event_indicator: int

    def __post_init__(self):
        self._is_fit, self.survival_function = self._fit_procedure()

    def estimate(self) -> Callable:
        if self._is_fit and self.survival_function is not None:
            return self.survival_function
        raise Exception("Unable to initialize KM function")

    def predict(self, t) -> Union[np.ndarray, pd.DataFrame]:
        if not self._is_fit:
            raise Exception("KM function not fit prior to prediction.")
        return self.survival_function(t)

    def _fit_procedure(self) -> Tuple[bool, Callable]:
        # Sort in case of batched data
        unique_event_times = np.sort(np.unique(
            self.time[self.delta == self.event_indicator]
        ))

        def hazard_function(t): return np.sum(
            self.weights * (self.time * self.delta == t)) / np.sum(self.weights * (self.time >= t))

        def greenwood_accumulation(t):
            di = np.sum(self.weights * (self.time * self.delta == t))
            ni = np.sum(self.weights * (self.time >= t))
            if ni == di:
                return 0
            return di / (ni * (ni - di))

        hazard = np.vectorize(hazard_function)(unique_event_times)
        var_hazard = np.vectorize(greenwood_accumulation)(unique_event_times)

        survival_probability = np.cumprod(1 - hazard)

        greenwood_formula = np.square(
            survival_probability) * np.cumsum(var_hazard)

        survival_estimates = np.vstack(
            ([0, 1, 0],
             np.column_stack(
                (unique_event_times, survival_probability, greenwood_formula)))
        )

        def survival_estimate(t: Union[np.ndarray, float]) -> np.ndarray:
            """Returns the time, survival estimate, and variance for the observed time. 

            Args:
                t (np.typing.ArrayLike): the time to evaluate survival at. 
            """

            # Note: np.searchsorted using "right" would return index 1 too large, but this does not include the first row.
            survival_indices = np.searchsorted(
                unique_event_times, t, side="right")

            return survival_estimates[survival_indices, :]

        self._is_fit = True

        return self._is_fit, survival_estimate
