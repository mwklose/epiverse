import pandas as pd
import numpy as np
from scipy.special import expit
from icecream import ic


class DataGeneratorWen():

    def __init__(self, n: int = 1000, time_steps: int = 5, seed: int = 700):
        self.rng = np.random.default_rng(seed=seed)
        self.data = pd.DataFrame({
            "L10": self.rng.binomial(1, p=0.5, size=n),
            "L20": self.rng.normal(2, 1, size=n),
            "Y0": np.zeros(n),    # Placeholder for using in loop below,
            "C0": np.zeros(n)     # Placeholder for using in loop below,
        })

        # Baseline
        self.data["A0"] = self.rng.binomial(
            n=1,
            p=expit(-1 + self.data["L10"] - 0.25 * self.data["L20"]),
            size=n)

        # Recurring
        for i in range(1, time_steps+1):
            # If not previously censored, then censor.
            possible_cj = self.rng.binomial(
                n=1,
                p=expit(-3 - 1 * np.nan_to_num(self.data[f"A{i-1}"]) + 0.75 *
                        np.nan_to_num(self.data[f"L1{i-1}"]) - 0.5 * np.nan_to_num(self.data[f"L2{i-1}"])),
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
                p=expit(-1 - 2 * np.nan_to_num(self.data[f"A{i-1}"]) + 2 *
                        np.nan_to_num(self.data[f"L1{i-1}"]) - 0.5 * np.nan_to_num(self.data[f"L2{i-1}"])),
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

            # If not had AIDS previously, then have AIDS now.
            possible_l1j = self.rng.binomial(
                n=1,
                p=expit(-2 - 2 *
                        np.nan_to_num(self.data[f"A{i-1}"]) - 0.5 * np.nan_to_num(self.data[f"L2{i-1}"]))
            )

            self.data[f"L1{i}"] = np.select(
                condlist=[
                    (self.data[f"Y{i}"] == 0) & (self.data[f"L1{i-1}"] == 0),
                    (self.data[f"Y{i}"] == 0) & (self.data[f"L1{i-1}"] == 1),
                    True
                ],
                choicelist=[
                    possible_l1j,
                    1,
                    np.nan
                ]
            )

            # Transformed CD4 count changing no matter what.
            possible_l2j = self.rng.normal(
                loc=2 + self.data[f"A{i-1}"] -
                self.data[f"L1{i-1}"] + 0.5 * self.data[f"L2{i-1}"],
                scale=1,
                size=n)

            self.data[f"L2{i}"] = np.select(
                condlist=[
                    self.data[f"Y{i}"] == 0,
                    True
                ],
                choicelist=[
                    possible_l2j,
                    np.nan
                ]
            )

            # If treated previously, then treated now.
            possible_aj = self.rng.binomial(
                n=1,
                p=expit(-1 + np.nan_to_num(self.data[f"L1{i}"]) -
                        0.25 * np.nan_to_num(self.data[f"L2{i}"])),
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
