from multiprocessing import Pool
from typing import Callable
import pandas as pd
import numpy as np


DEFAULT_ITERATIONS = 10000


class Bootstrap:
    # Kwargs must be in order of
    def __init__(self, data: pd.DataFrame, bootstrap_function: Callable, number_in_sample=None, number_of_iterations=None, **kwargs):
        self.data = data
        self.bootstrap_function = bootstrap_function

        self.data_size = self.data.shape[0]
        self.number_in_sample = self.data.shape[0] if not number_in_sample else number_in_sample
        self.number_of_iterations = DEFAULT_ITERATIONS if not number_of_iterations else number_of_iterations

        self.results = self.run_bootstrap(**kwargs)

    def run_bootstrap(self, **kwargs):
        rng = np.random.default_rng()
        index_samples = rng.integers(low=0, high=self.data_size, size=(
            self.number_of_iterations, self.number_in_sample))

        def test_fun(x):
            return self.data.iloc[x]

        with Pool() as pool:
            data_samples = np.apply_along_axis(
                test_fun, axis=1, arr=index_samples)

            data_samples_and_kwargs = [(pd.DataFrame(data, columns=self.data.columns),
                                        *kwargs.values())
                                       for data in data_samples]
            results = pool.starmap(
                self.bootstrap_function, data_samples_and_kwargs)
        results = pd.DataFrame(results)
        return results
