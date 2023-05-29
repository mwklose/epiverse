from typing import List
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from epiverse.utilities.data_generation.data_generator import DataGenerator


class DataGeneratorBinaryConfoundingTriangle(DataGenerator):

    def __init__(self, n: int = 100, treatment_prevalence: float = None, confounder_prevalence: float = None,
                 outcome_prevalence: float = None, odds_ratio: float = None, confounder_odds_ratio: float = 1.5,
                 y1: float = None, y0: float = None):

        if n < 0:
            raise Exception("Must have positive number of observations.")
        self.n = n

        if treatment_prevalence is None:
            raise Exception("Treatment prevalence must not be none")
        self.treatment_prevalence = treatment_prevalence

        if confounder_prevalence is None:
            raise Exception("Confounder prevalence must not be none")
        self.confounder_prevalence = confounder_prevalence

        # Split logic into two different generating functions
        self.generate_by_counterfactuals = self.check_y(y1, y0)

        if not self.generate_by_counterfactuals:
            self.check_odds_ratio(
                odds_ratio, outcome_prevalence, confounder_odds_ratio)

    def check_y(self, y1: float, y0: float):
        if y1 is None and y0 is None:
            return False

        if 0 > y1 or y1 > 1:
            raise Exception("Y1 must be on interval [0,1]")
        if 0 > y0 or y0 > 1:
            raise Exception("Y0 must be on interval [0,1]")

        self.y1 = y1
        self.y0 = y0

        return True

    def check_odds_ratio(self, odds_ratio: float, outcome_prevalence: float, confounder_odds_ratio: float):
        if outcome_prevalence is None:
            raise Exception("Outcome prevalence must not be none")
        if 0 > outcome_prevalence or 1 < outcome_prevalence:
            raise Exception("Outcome prevalence must be on interval [0,1]")
        self.outcome_prevalence = outcome_prevalence

        if odds_ratio <= 0:
            raise Exception("Odds ratio must be on interval (0, ∞)")
        self.odds_ratio = odds_ratio

        if confounder_odds_ratio <= 0:
            raise Exception("Confounder odds ratio must be on interval (0, ∞)")
        self.confounder_odds_ratio = confounder_odds_ratio

    def generate_data(self) -> pd.DataFrame:
        # Generates Data for a 3 node confounding triangle
        rng = np.random.default_rng()
        # Generate W
        w = rng.binomial(1, self.confounder_prevalence, self.n)

        # Generate A from that
        a_prevalence = logit(self.treatment_prevalence) - (np.average(w) - w)
        a = rng.binomial(1, expit(a_prevalence), self.n)

        return self.generate_data_by_counterfactuals(rng, w, a) \
            if self.generate_by_counterfactuals \
            else self.generate_data_by_odds_ratio(rng, w, a)

    def generate_data_by_counterfactuals(self, rng: np.random.Generator, w: np.array, a: np.array):

        w_mean = pd.DataFrame({"W": w, "A": a}).groupby(
            ["A"]).mean().to_numpy()

        y0_prevalence = logit(self.y0) - (w_mean[0] - w)
        y0 = rng.binomial(1, expit(y0_prevalence), self.n)

        y1_prevalence = logit(self.y1) - (w_mean[1] - w)
        y1 = rng.binomial(1, expit(y1_prevalence), self.n)

        y_observed = np.where(a == 1, y1, y0)

        return pd.DataFrame({
            "W": w,
            "A": a,
            "Y0": y0,
            "Y1": y1,
            "Y": y_observed
        })

    def generate_data_by_odds_ratio(self, rng: np.random.Generator, w: np.array, a: np.array):
        # Generate Y from W and A
        y_prevalence = logit(self.outcome_prevalence) \
            - np.log(self.confounder_odds_ratio) * (np.average(w) - w) - \
            np.log(self.odds_ratio) * (np.average(a) - a)
        y = rng.binomial(1, expit(y_prevalence), self.n)

        # Return a DataFrame
        return pd.DataFrame({"W": w,
                             "A": a,
                             "Y": y})
