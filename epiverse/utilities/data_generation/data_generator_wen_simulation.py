

import pandas as pd
import numpy as np
from scipy.special import expit
from icecream import ic


class DataGeneratorWenSimulation():

    def __init__(self, n: int = 1000, time_steps: int = 5, seed: int = 700):
        self.rng = np.random.default_rng(seed=seed)
        self.data = pd.DataFrame({
            "L0": self.rng.binomial(1, p=0.5, size=n),
            # Placeholders for iterative definition.
            "Y0": 0,
            "C0": 0
        })

        # Baseline
        self.data["A0"] = self.rng.binomial(
            n=1,
            p=expit(-1.5 + self.data["L0"]),
            size=n)

        # Recurring
        for i in range(1, time_steps+1):
            # If not previously censored, then censor.
            possible_cj = self.rng.binomial(
                n=1,
                p=expit(-3 - 1 * np.nan_to_num(self.data[f"A{i-1}"]) + 0.75 *
                        np.nan_to_num(self.data[f"L{i-1}"])),
                size=n)
            self.data[f"C{i}"] = np.select(
                condlist=[
                    self.data[f"Y{i-1}"] == 0,
                    self.data[f"C{i-1}"] == 1,
                    True
                ],
                choicelist=[
                    possible_cj,
                    1,
                    np.nan
                ]
            )

            # If had event previously, then have event now.
            possible_yj = self.rng.binomial(
                n=1,
                p=expit(-2 - 2 * np.nan_to_num(self.data[f"A{i-1}"]) +
                        np.nan_to_num(self.data[f"L{i-1}"])),
                size=n)

            self.data[f"Y{i}"] = np.select(
                condlist=[
                    (self.data[f"Y{i-1}"] == 0) & (self.data[f"C{i}"] == 0),
                    self.data[f"Y{i-1}"] == 1,
                    True
                ],
                choicelist=[
                    possible_yj,
                    1,
                    np.nan
                ]
            )

            # If not had AIDS before, then may have AIDS now.
            possible_lj = self.rng.binomial(
                n=1,
                p=expit(-2 - 2 * np.nan_to_num(self.data[f"A{i-1}"]))
            )

            self.data[f"L{i}"] = np.select(
                condlist=[
                    (self.data[f"Y{i}"] == 0) & (self.data[f"L{i-1}"] == 0),
                    (self.data[f"Y{i}"] == 0) & (self.data[f"L{i-1}"] == 1),
                    True
                ],
                choicelist=[
                    possible_lj,
                    1,
                    np.nan
                ]
            )

            # If treated before, then treated now.
            possible_aj = self.rng.binomial(
                n=1,
                p=expit(-1.5 + np.nan_to_num(self.data[f"L{i}"])),
                size=n
            )

            self.data[f"A{i}"] = np.select(
                condlist=[
                    (self.data[f"Y{i}"] == 0) & (self.data[f"A{i-1}"] == 0),
                    (self.data[f"Y{i}"] == 0) & (self.data[f"A{i-1}"] == 1),
                    True
                ],
                choicelist=[
                    possible_aj,
                    1,
                    np.nan
                ]
            )
