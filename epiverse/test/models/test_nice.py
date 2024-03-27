from epiverse.models.noniterative_conditional_expectation import noniterative_ice, noniterative_ice_constructor
from epiverse.models import MultioutcomeRegression, LogisticRegression
from epiverse.utilities.data_generation import DataGeneratorWen, DataGeneratorWenSimulation
from icecream import ic
import statsmodels.api as sm
import numpy as np


def test_nice_constructor():
    TIME_STEPS = 5
    N = 1000
    mydata = DataGeneratorWen(n=N, time_steps=TIME_STEPS)
    mydata.data["id"] = range(N)

    ld = noniterative_ice_constructor(
        lambda t, ls: 1,
        confounder_list=[[f"L1{k}", f"L2{k}"] for k in range(TIME_STEPS+1)],
        treatment_list=[f"A{k}" for k in range(TIME_STEPS+1)],
        outcome_list=[f"Y{k}" for k in range(1, TIME_STEPS+1)],
        censoring_list=[f"C{k}" for k in range(1, TIME_STEPS+1)],
        data=mydata.data,
        pooled_confounder_model=lambda eqn, data: MultioutcomeRegression(
            eqn=eqn, data=data, family_list=[sm.families.Binomial(), sm.families.Gaussian()]),
        pooled_outcome_model=lambda eqn, data: LogisticRegression(
            eqn=eqn, data=data),
        id_col="id",
        treatment_prefix="A",
        confounder_prefixes=["L1", "L2"],
        censoring_prefix="C",
        outcome_prefix="Y",
        batch_size=1000
    )

    ic(ld)
    assert np.all(ld > 0)
    assert np.all(ld < 1)


def test_nice_unique_regime():
    TIME_STEPS = 5
    N = 5000
    mydata = DataGeneratorWen(n=N, time_steps=TIME_STEPS)
    mydata.data["id"] = range(N)

    def treatment_regime_e21(t, ls):
        l1 = ls[ls.columns[1]].to_numpy()
        l2 = ls[ls.columns[2]].to_numpy()
        return np.where((l2 < 2.75) | (l1 == 1), np.ones(ls.shape[0]), np.zeros(ls.shape[0]))

    ld = noniterative_ice_constructor(
        treatment_strategy=treatment_regime_e21,
        confounder_list=[[f"L1{k}", f"L2{k}"] for k in range(TIME_STEPS+1)],
        treatment_list=[f"A{k}" for k in range(TIME_STEPS+1)],
        outcome_list=[f"Y{k}" for k in range(1, TIME_STEPS+1)],
        censoring_list=[f"C{k}" for k in range(1, TIME_STEPS+1)],
        data=mydata.data,
        pooled_confounder_model=lambda eqn, data: MultioutcomeRegression(
            eqn=eqn, data=data, family_list=[sm.families.Binomial(), sm.families.Gaussian()]),
        pooled_outcome_model=lambda eqn, data: LogisticRegression(
            eqn=eqn, data=data),
        id_col="id",
        treatment_prefix="A",
        confounder_prefixes=["L1", "L2"],
        censoring_prefix="C",
        outcome_prefix="Y",
        batch_size=1000
    )

    ic(ld)

    assert np.all(ld > 0)
    assert np.all(ld < 1)


def test_wen_simulation():
    TIME_STEPS = 5
    N = 5000
    mydata = DataGeneratorWenSimulation(n=N, time_steps=TIME_STEPS)
    mydata.data["id"] = range(N)

    ld = noniterative_ice_constructor(
        lambda t, ls: 1,
        confounder_list=[[f"L{k}"] for k in range(TIME_STEPS+1)],
        treatment_list=[f"A{k}" for k in range(TIME_STEPS+1)],
        outcome_list=[f"Y{k}" for k in range(1, TIME_STEPS+1)],
        censoring_list=[f"C{k}" for k in range(1, TIME_STEPS+1)],
        data=mydata.data,
        pooled_confounder_model=lambda eqn, data: LogisticRegression(
            eqn=eqn, data=data),
        pooled_outcome_model=lambda eqn, data: LogisticRegression(
            eqn=eqn, data=data),
        id_col="id",
        treatment_prefix="A",
        confounder_prefixes=["L"],
        censoring_prefix="C",
        outcome_prefix="Y",
        batch_size=1000
    )
