from epiverse.survival.turnbull import Turnbull
from epiverse.survival.kaplan_meier import KaplanMeier
from epiverse.utilities.data_generation import DataGeneratorTurnbull

import numpy as np
import pandas as pd
from icecream import ic

def test_turnbull(): 
    rng: np.random.Generator = np.random.default_rng(seed=776)
    
    def integer_truncation(n) -> np.ndarray: 
        low: np.ndarray = rng.integers(low=4, high=20, size=n)
        high: np.ndarray = np.full(shape=low.shape, fill_value=np.inf)
        return np.column_stack(tup=[low, high])

    def integer_censoring(n) -> np.ndarray: 
        value: np.ndarray = rng.uniform(low=1, high=45, size=n)
        return value
    
    tb = DataGeneratorTurnbull(
        truncation_generator=integer_truncation,
        censoring_generator=integer_censoring,
        n=100
    )

    turnbull = Turnbull(
        data=tb.generate_data()
    )

    assert np.abs(np.sum(turnbull.s) - 1.0) < 1e-9


def test_turnbull_km(): 
    data = pd.DataFrame({
        "id": range(6), 
        "indicator": [1,0,1,0,1,1], 
        "censoring_low": [3, 4, 5, 7, 9, 10]
    })

    data["censoring_high"] = np.where(data["indicator"] == 1, data["censoring_low"], np.inf)
    data["truncation_low"] = 0
    data["truncation_high"] = np.inf

    ic(data)
    km = KaplanMeier(time=data["censoring_low"], delta=data["indicator"], weights=np.ones(len(data["indicator"])), event_indicator=1)

    tb = Turnbull(data=data)

    ic(km.survival_estimates)
    ic(tb.survival_estimates)