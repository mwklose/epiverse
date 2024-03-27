from numpy import full
from epiverse.utilities.data_generation import DataGenerator
from dataclasses import dataclass
from typing import Callable, Tuple
import pandas as pd
import numpy as np

from icecream import ic


@dataclass
class DataGeneratorTurnbull(DataGenerator): 
    truncation_generator: Callable 
    censoring_generator: Callable
    n: int = 100
    seed: int = 776
    
    def __post_init__(self) -> None: 
        self.rng = np.random.default_rng(seed=self.seed)
        self.unseen_data: pd.DataFrame
        self.seen_data: pd.DataFrame
        self.unseen_data, self.seen_data = self._seen_unseen_data()
        

    def _seen_unseen_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]: 
        full_data = pd.DataFrame(data={'id': range(0, self.n, 1)})
        full_data["censoring_low"] = self.censoring_generator(self.n)
        full_data["censoring_high"] = full_data["censoring_low"]
        full_data["censoring_high"] = np.where(self.rng.binomial(n=1, p=0.1, size=self.n) == 1, np.inf, full_data["censoring_high"])
        full_data[["truncation_low", "truncation_high"]] = self.truncation_generator(self.n)

        is_seen: pd.Series[bool] = full_data["censoring_low"].between(left=full_data["truncation_low"], right=full_data["truncation_high"]) & full_data["censoring_high"].between(left=full_data["truncation_low"], right=full_data["truncation_high"]) 

        return full_data[~is_seen], full_data[is_seen]

    def generate_data(self) -> pd.DataFrame:
        return self.seen_data