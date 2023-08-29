import numpy as np
import pandas as pd

from epiverse.models.gcomputation import GComputation
from epiverse.models.discrete_density import DiscreteDensity
from epiverse.models.logistic_regression import LogisticRegression

SIMULATED_DATA_PATH = "epiverse/data/data.csv"

df = pd.read_csv(SIMULATED_DATA_PATH)

gc = GComputation(
    outcome_model=LogisticRegression(),
    density_models=[
        DiscreteDensity(W0=df["W0"]),
        DiscreteDensity(W1=df["W1"],
                        W0=df["W0"], A0=df["A0"]),
        DiscreteDensity(W2=df["W2"], W1=df["W1"], W0=df["W0"],
                        A1=df["A1"], A0=df["A0"])
    ],
).fit(
    data=df,
    outcome_column="Y3",
    treatment_columns=["A0", "A1", "A2"],
    covariate_columns=[["W0"], ["W1"], ["W2"]],
    covariate_conditioning_sets=[["A0", "W0"],
                                 ["A1", "A0", "W1", "W0"],
                                 ["A2", "A1", "W2", "W1"]]
)


def always_treat_func(t, row):
    return 1


def never_treat_func(t, row):
    return 0


def as_treated_func(t, row):
    return row[t]


def counterfactual_func(t, row):
    return 1 - row[t]


always_treat = gc.predict(always_treat_func)
never_treat = gc.predict(never_treat_func)
as_treated = gc.predict(as_treated_func)
counterfactual = gc.predict(counterfactual_func)

print(f"""\talways: {always_treat}, 
      never: {never_treat}
      as treated: {as_treated}, 
      counterfactual: {counterfactual}""")
