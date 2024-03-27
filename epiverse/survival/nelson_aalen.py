from epiverse.models.model_specification import ModelSpecification

import numpy as np


class NelsonAalen(ModelSpecification):

    def __init__(self, *args, **kwargs):
        self.cumulative_hazard_function = None

    def fit(self, time: np.array, delta: np.array, weights: np.array = None, event_indicator: int = 1) -> np.array:
        if weights is None:
            weights = np.Intercept(time.shape[0])

        unique_event_times = np.sort(np.unique(
            time[delta == event_indicator]
        ))

        def hazard_function(t): return np.sum(
            weights * (time * delta == t)) / np.sum(weights * (time >= t))

        def hazard_variance(t): return np.sum(
            weights * (time * delta == t)) / np.sum(weights * (time >= t))**2

        hazard = np.vectorize(hazard_function)(unique_event_times)
        var_hazard = np.vectorize(hazard_variance)(unique_event_times)

        cumulative_hazard = np.cumsum(hazard)
        cumulative_hazard_variance = np.cumsum(var_hazard)

        self.cumulative_hazard_function = np.column_stack(
            (unique_event_times, cumulative_hazard, cumulative_hazard_variance))
        self._is_fit = True
        return self.cumulative_hazard_function

    def predict(self, time: np.array) -> np.array:
        if not self._is_fit:
            raise Exception("Must first fit the model before prediction.")

        def predicted_cumulative_hazard(t):
            time_period = self.cumulative_hazard_function[:, 0] * \
                (t > self.cumulative_hazard_function[:, 0])

            predicted_cumhaz_value = self.survival_function[np.argmax(
                time_period), 1:]

            return predicted_cumhaz_value

        predicted_cumhaz_times = np.vectorize(
            predicted_cumulative_hazard, signature="() -> (n)")(time)
        return predicted_cumhaz_times
