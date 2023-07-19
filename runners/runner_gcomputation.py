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

always_treat = gc.predict([1, 1, 1])
never_treat = gc.predict([0, 0, 0])
natural_course = gc.predict([
    df["A0"],
    df["A1"],
    df["A2"]
])

print(
    f"Always treat: {always_treat}, never treat: {never_treat}, natural course: {natural_course}")
