import numpy as np
import pandas as pd

from epiverse.models.ipw_marginal_structural_model import IPWMarginalStructuralModel
from epiverse.models.discrete_density import DiscreteDensity
SIMULATED_DATA_PATH = "epiverse/data/data.csv"

df = pd.read_csv(SIMULATED_DATA_PATH)

ipw = IPWMarginalStructuralModel(
    density_models=[
        DiscreteDensity(A0=df["A0"], W0=df["W0"]),
        DiscreteDensity(A1=df["A1"], W1=df["W1"],
                        A0=df["A0"], W0=df["W0"]),
        DiscreteDensity(A2=df["A2"], W2=df["W2"],
                        A1=df["A1"], W1=df["W1"],
                        A0=df["A0"], W0=df["W0"])
    ],
    stablized_constructor=DiscreteDensity
).fit(
    data=df,
    outcome_column="Y3",
    treatment_columns=["A0", "A1", "A2"],
    covariate_columns=[["W0"], ["W1"], ["W2"]]
)

always_treat = ipw.predict([1, 1, 1])
never_treat = ipw.predict([0, 0, 0])
natural_course = ipw.predict([
    df["A0"],
    df["A1"],
    df["A2"]
])

print(
    f"Always treat: {always_treat}, never treat: {never_treat}, natural course: {natural_course}")
