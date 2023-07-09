import numpy as np
import pandas as pd

from epiverse.models.gcomputation import GComputation
from epiverse.models.discrete_density import DiscreteDensity
SIMULATED_DATA_PATH = "epiverse/data/data.csv"

df = pd.read_csv(SIMULATED_DATA_PATH)

print(df["W0"])


GComputation(
    outcome_model=[
        DiscreteDensity(data=df)
    ],
    density_models=[
        DiscreteDensity(W0=df["W0"]).fit(),
        DiscreteDensity(W1=df["W1"],
                        W0=df["W0"], A0=df["A0"]).fit(),
        DiscreteDensity(W2=df["W2"], W1=df["W1"], W0=df["W0"],
                        A1=df["A1"], A0=df["A0"]).fit()
    ],
).fit(
    outcome_column="Y3",
    treatment_columns=["A0", "A1", "A2"],
    covariate_columns=[["W0"], ["W1"], ["W2"]],
    covariate_conditioning_sets=[["A0", "W0"],
                                 ["A1", "A0", "W1", "W0"],
                                 ["A2", "A1", "W2", "W1"]]
)
