from epiverse.utilities.data_generation import DataGeneratorWen, DataGeneratorWenSimulation
from epiverse.models.noniterative_conditional_expectation import noniterative_ice_constructor
from epiverse.models.stratified_iterative_conditional_expectation import stratified_ice_constructor
from epiverse.models import LogisticRegression, PooledMultioutcomeRegression, PooledLogisticRegression
from icecream import ic
import statsmodels.api as sm


def test_simulation1():
    TIME_STEPS = 5
    N = 10000
    mydata = DataGeneratorWen(n=N, time_steps=TIME_STEPS)
    mydata.data["id"] = range(N)

    def treatment_regime(t, ls): return 1

    ld1 = noniterative_ice_constructor(
        treatment_strategy=treatment_regime,
        confounder_list=[[f"L1{k}", f"L2{k}"] for k in range(TIME_STEPS+1)],
        treatment_list=[f"A{k}" for k in range(TIME_STEPS+1)],
        outcome_list=[f"Y{k}" for k in range(1, TIME_STEPS+1)],
        censoring_list=[f"C{k}" for k in range(1, TIME_STEPS+1)],
        data=mydata.data,
        pooled_confounder_model=lambda eqn, data: PooledMultioutcomeRegression(
            eqn=eqn, time_var="t", data=data, family_list=[sm.families.Binomial(), sm.families.Gaussian()]),
        pooled_outcome_model=lambda eqn, data: PooledLogisticRegression(
            eqn=eqn, time_var="t", data=data),
        id_col="id",
        treatment_prefix="A",
        confounder_prefixes=["L1", "L2"],
        censoring_prefix="C",
        outcome_prefix="Y",
        batch_size=1000,
        tol=1e-7
    )

    ld2 = stratified_ice_constructor(treatment_regime,
                                     confounder_list=[
                                         [f"L1{k}", f"L2{k}"] for k in range(TIME_STEPS+1)],
                                     treatment_list=[
                                         f"A{k}" for k in range(TIME_STEPS+1)],
                                     outcome_list=[
                                         f"Y{k}" for k in range(1, TIME_STEPS+1)],
                                     censoring_list=[
                                         f"C{k}" for k in range(1, TIME_STEPS+1)],
                                     data=mydata.data,
                                     outcome_prefix="Y")

    ic(ld1)
    ic(ld2)


def test_simulation2():
    TIME_STEPS = 5
    N = 10000
    mydata = DataGeneratorWenSimulation(n=N, time_steps=TIME_STEPS)
    mydata.data["id"] = range(N)

    def treatment_regime(t, ls): return 1

    ld1 = noniterative_ice_constructor(
        treatment_strategy=treatment_regime,
        confounder_list=[[f"L{k}"] for k in range(TIME_STEPS+1)],
        treatment_list=[f"A{k}" for k in range(TIME_STEPS+1)],
        outcome_list=[f"Y{k}" for k in range(1, TIME_STEPS+1)],
        censoring_list=[f"C{k}" for k in range(1, TIME_STEPS+1)],
        data=mydata.data,
        pooled_confounder_model=lambda eqn, data: PooledLogisticRegression(
            eqn=eqn, time_var="t", data=data),
        pooled_outcome_model=lambda eqn, data: PooledLogisticRegression(
            eqn=eqn, time_var="t", data=data),
        id_col="id",
        treatment_prefix="A",
        confounder_prefixes=["L"],
        censoring_prefix="C",
        outcome_prefix="Y",
        batch_size=1000
    )

    ld2 = stratified_ice_constructor(treatment_regime,
                                     confounder_list=[
                                         [f"L{k}"] for k in range(TIME_STEPS+1)],
                                     treatment_list=[
                                         f"A{k}" for k in range(TIME_STEPS+1)],
                                     outcome_list=[
                                         f"Y{k}" for k in range(1, TIME_STEPS+1)],
                                     censoring_list=[
                                         f"C{k}" for k in range(1, TIME_STEPS+1)],
                                     data=mydata.data,
                                     outcome_prefix="Y")

    ic(ld1)
    ic(ld2)
