import numpy as np
import pandas as pd

from epiverse.models import FunctionModel
from epiverse.survival import KaplanMeier
from dataclasses import dataclass
from typing import Union, Callable, Tuple, List


@dataclass
class AalenJohansen(FunctionModel):
    time: Union[np.ndarray, pd.Series]
    delta: Union[np.ndarray, pd.Series]
    weights: Union[np.ndarray, pd.Series]
    event_indicator: List[int]

    def __post_init__(self):
        self.unique_indicators: int = np.unique(self.delta)
        self._is_fit, self.incidence_function = self._fit_procedure()

    def estimate(self, cause: int = None) -> Callable:
        if not self._is_fit:
            raise Exception("Fitting process for AalenJohansen not completed.")
        if cause is None:
            raise Exception("Must choose cause-specific incidence function.")

        if not cause in self.unique_indicators:
            raise Exception("Cause not found in AJ function.")

        return lambda t: self.incidence_function(t, cause)

    def predict(self, t, cause: int = None) -> Union[np.ndarray, pd.DataFrame]:
        if not self._is_fit:
            raise Exception("AJ function not fit prior to prediction.")

        if not cause in self.unique_indicators:
            raise Exception("Cause not found in AJ function for prediction.")

        return self.incidence_function(t, cause)

    def _fit_procedure(self) -> Tuple[bool, List[Callable]]:
        # Sort in case of batched data
        unique_event_times = np.sort(np.unique(
            self.time[np.in1d(self.delta, self.event_indicator)]
        ))

        # Aalen Johansen fit process:
        # 1. Overall survival
        overall_survival = KaplanMeier(self.time,
                                       np.in1d(
                                           self.delta, self.event_indicator),
                                       weights=self.weights,
                                       event_indicator=1)

        survival_estimates = np.roll(
            overall_survival.predict(unique_event_times), 1, axis=0)
        survival_estimates[0, 1:] = 1
        survival_estimates[:, 0] = unique_event_times

        print(f"survival:\n{survival_estimates}")

        # 2. Cause-specific hazards at each time point

        def cause_specific_hazard(t, cause):
            return np.sum(
                self.weights * ((self.time == t) * (self.delta == cause))) / np.sum(self.weights * (self.time >= t))

        # 3. Accumulation of cause-specific hazards
        cause_specific_hazards = np.zeros(
            (unique_event_times.shape[0], len(self.event_indicator)))

        for i, ind in enumerate(self.event_indicator):
            cause_specific_hazards[:, i] += np.vectorize(
                lambda t: cause_specific_hazard(t, ind))(unique_event_times)

        piecewise_incidence = cause_specific_hazards * \
            survival_estimates[:, 1][:, None]

        cause_specific_incidence = np.cumsum(piecewise_incidence, axis=0)

        # TODO: variance

        def greenwood_accumulation(t):
            di = np.sum(self.weights * (self.time * self.delta == t))
            ni = np.sum(self.weights * (self.time >= t))
            if ni == di:
                return 0
            return di / (ni * (ni - di))

        # var_hazard = np.vectorize(greenwood_accumulation)(unique_event_times)

        # survival_probability = np.cumprod(1 - hazard)

        # greenwood_formula = np.square(
        #     survival_probability) * np.cumsum(var_hazard)

        # 4. Create function

        incidence_estimates = np.vstack((
            [0] * (len(self.event_indicator) + 1),
            np.column_stack((unique_event_times, cause_specific_incidence))
        ))

        print(f"incidence est:\n{incidence_estimates}")

        def incidence_estimate(t: Union[np.ndarray, float], cause: int) -> np.ndarray:
            """Returns the time, survival estimate, and variance for the observed time. 

            Args:
                t (np.typing.ArrayLike): the time to evaluate survival at. 
            """

            # Note: np.searchsorted using "right" would return index 1 too large, but this does not include the first row.
            incidence_indices = np.searchsorted(
                unique_event_times, t, side="right")

            return incidence_estimates[incidence_indices, cause]

        self._is_fit = True

        return self._is_fit, incidence_estimate
