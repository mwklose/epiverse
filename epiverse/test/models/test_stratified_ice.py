from epiverse.models import stratified_ice, stratified_ice_constructor
from epiverse.utilities.data_generation import DataGeneratorWenSimulation
from icecream import ic
import statsmodels.api as sm
import numpy as np


def test_stratified_ice():
    TIME_STEPS = 5
    N = 1000
    mydata = DataGeneratorWenSimulation(n=N, time_steps=TIME_STEPS)
    mydata.data["id"] = range(N)

    ic(mydata.data.columns)
    ld = stratified_ice_constructor(
        lambda t, ls: 1,
        confounder_list=[[f"L{t}"] for t in range(TIME_STEPS)],
        treatment_list=[f"A{t}" for t in range(TIME_STEPS)],
        outcome_list=[f"Y{t+1}" for t in range(TIME_STEPS)],
        censoring_list=[f"C{t+1}" for t in range(TIME_STEPS)],
        data=mydata.data,
        outcome_prefix="Y"
    )

    ic(ld)

    assert False
