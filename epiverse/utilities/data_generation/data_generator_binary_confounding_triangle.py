from typing import List
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from epiverse.utilities.data_generation.data_generator import DataGenerator


class DataGeneratorBinaryConfoundingTriangle(DataGenerator):

    def __init__(self, n: int = 100, treatment_prevalence: float = None, confounder_prevalence: float = None, outcome_prevalence: float = None, odds_ratio: float = None):
        self.n = n

        if treatment_prevalence is None:
            raise Exception("Treatment prevalence must not be none")
        self.treatment_prevalence = treatment_prevalence

        if confounder_prevalence is None:
            raise Exception("Confounder prevalence must not be none")
        self.confounder_prevalence = confounder_prevalence

        if outcome_prevalence is None:
            raise Exception("Outcome prevalence must not be none")
        self.outcome_prevalence = outcome_prevalence

        if odds_ratio <= 0:
            raise Exception("Odds ratio must be on interval (0, ∞)")
        self.odds_ratio = odds_ratio

    def generate_data(self) -> pd.DataFrame:
        # Generates Data for a 3 node confounding triangle
        rng = np.random.default_rng()
        # Generate W
        w = rng.binomial(1, self.confounder_prevalence, self.n)

        # Generate A from that
        a_prevalence = logit(self.treatment_prevalence) - (np.average(w) - w)
        a = rng.binomial(1, expit(a_prevalence), self.n)

        # Generate Y from W and A
        y_prevalence = logit(self.outcome_prevalence) \
            - np.log(1.5) * (np.average(w) - w) - \
            np.log(self.odds_ratio) * (np.average(a) - a)
        y = rng.binomial(1, expit(y_prevalence), self.n)

        # Return a DataFrame
        return pd.DataFrame({"W": w,
                             "A": a,
                             "Y": y})
