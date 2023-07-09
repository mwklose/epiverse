from epiverse.models.model_specification import ModelSpecification
import numpy as np


class KaplanMeier(ModelSpecification):

    def __init__(self, *args, **kwargs):
        self.survival_function = None

    def fit(self, **kwargs):
        # Error checking
        if "time" not in kwargs:
            raise Exception("Numpy array of 'time' needs to be provided.")
        elif not isinstance(kwargs["time"], np.ndarray):
            raise Exception(
                f"Numpy array of 'time' needs to be provided, was {type(kwargs['time'])}")

        time = kwargs["time"]

        if "delta" not in kwargs:
            raise Exception(
                "Numpy array of 'delta' (event indicator) needs to be provided.")
        elif not isinstance(kwargs["delta"], np.ndarray):
            raise Exception(
                f"Numpy array of 'delta' (event indicator) needs to be provided, was {type(kwargs['delta'])}")
        elif time.shape != kwargs["delta"].shape:
            raise Exception(
                f"Shapes of time ({time.shape}) and delta ({kwargs['delta'].shape}) arrays are different.")

        delta = kwargs["delta"]

        if "weights" not in kwargs:
            weights = np.ones(time.shape[0])
        elif not isinstance(kwargs["weights"], np.ndarray):
            raise Exception(
                f"Weights must also be a numpy array, but is {type(kwargs['weights'])}")
        else:
            weights = kwargs["weights"]

        if "event_indicator" not in kwargs:
            event_indicator = 1
        else:
            event_indicator = kwargs["event_indicator"]

        # Sort in case of batched data
        unique_event_times = np.sort(np.unique(
            time[delta == event_indicator]
        ))

        def hazard_function(t): return np.sum(
            weights * (time * delta == t)) / np.sum(weights * (time >= t))

        def greenwood_accumulation(t):
            di = np.sum(weights * (time * delta == t))
            ni = np.sum(weights * (time >= t))
            if ni == di:
                return 0
            return di / (ni * (ni - di))

        hazard = np.vectorize(hazard_function)(unique_event_times)
        var_hazard = np.vectorize(greenwood_accumulation)(unique_event_times)

        survival_probability = np.cumprod(1 - hazard)

        greenwood_formula = np.square(
            survival_probability) * np.cumsum(var_hazard)

        self.survival_function = np.vstack(
            ([0, 1, 0],
             np.column_stack(
                (unique_event_times, survival_probability, greenwood_formula)))
        )

        self._is_fit = True

        return self.survival_function

    def predict(self, time: np.array):
        if not self._is_fit:
            raise Exception("Must first fit the model before prediction.")

        def predicted_survival(t):
            time_period = self.survival_function[:,
                                                 0] * (t > self.survival_function[:, 0])

            predicted_survival_value = self.survival_function[np.argmax(
                time_period), 1:]

            return predicted_survival_value

        predicted_survival_times = np.vectorize(
            predicted_survival, signature="() -> (n)")(time)
        return predicted_survival_times
