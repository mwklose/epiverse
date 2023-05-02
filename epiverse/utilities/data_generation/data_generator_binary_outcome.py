from epiverse.utilities.data_generation.data_generator_binary import DataGeneratorBinary
import numpy as np
from scipy.stats import binom
import pandas as pd


class DataGeneratorBinaryOutcomeTreatment(DataGeneratorBinary):
    def __init__(self, n: int = 10000, baseline_prevalence: float = None, treatment_prevalence: float = None, risk_difference: float = None, risk_ratio: float = None, odds_ratio: float = None):
        if treatment_prevalence is None:
            self.data = self.generate_data_by_effect_measure(
                n=n, baseline_prevalence=baseline_prevalence, risk_difference=risk_difference, risk_ratio=risk_ratio, odds_ratio=odds_ratio)
        else:
            self.data = self.generate_data_by_prevalence(
                n=n, baseline_prevalence=baseline_prevalence, treatment_prevalence=treatment_prevalence)

    def generate_data(self):
        pass

    def generate_data_by_prevalence(self, baseline_prevalence=None, treatment_prevalence=None):
        pass

    def generate_data_by_effect_measure(self, n: int, baseline_prevalence: float = None, risk_difference: float = None, risk_ratio: float = None, odds_ratio: float = None):
        measures = self.check_effect_measures(baseline_prevalence=baseline_prevalence,
                                              risk_difference=risk_difference, risk_ratio=risk_ratio, odds_ratio=odds_ratio)
        n_baseline = np.floor(n / 2).astype(int)
        n_treatment = n - n_baseline

        baseline_exposure = np.zeros((n_baseline, 1))
        baseline_outcome = binom.rvs(
            n=1, p=baseline_prevalence, size=n_baseline)
        baseline_outcome = np.reshape(baseline_outcome, (n_baseline, 1))

        treated_exposure = np.ones((n_treatment, 1))
        treated_outcome = binom.rvs(
            n=1, p=baseline_prevalence + measures["RD"], size=n_treatment
        )
        treated_outcome = np.reshape(treated_outcome, (n_treatment, 1))

        baseline = np.hstack((baseline_exposure, baseline_outcome))
        treated = np.hstack((treated_exposure, treated_outcome))
        combined_dataset = np.vstack((baseline, treated))
        return pd.DataFrame(combined_dataset, columns=["Exposure", "Outcome"])
