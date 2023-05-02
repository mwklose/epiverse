from abc import ABC, abstractmethod
from epiverse.utilities.data_generation.data_generator import DataGenerator
from typing import Dict


class DataGeneratorBinary(DataGenerator, ABC):

    @abstractmethod
    def generate_data_by_prevalence(self):
        pass

    @abstractmethod
    def generate_data_by_effect_measure(self):
        pass

    def check_effect_measures(self, baseline_prevalence: float = None, risk_difference: float = None, risk_ratio: float = None, odds_ratio: float = None) -> Dict:
        if baseline_prevalence is None:
            raise Exception(
                "Baseline prevalence for effect measure must not be None.")

        if risk_difference is None and risk_ratio is None and odds_ratio is None:
            raise Exception("One non-None effect measure must be provided.")

        if risk_difference is not None:
            if risk_difference < -1 or risk_difference > 1:
                raise Exception("Risk difference must be in [-1, 1].")
            if risk_ratio is not None or odds_ratio is not None:
                raise Exception("Only one effect measure must be provided.")

            treatment_prevalence = (baseline_prevalence + risk_difference)
            if treatment_prevalence > 1 or treatment_prevalence < -1:
                raise Exception(
                    f"Invalid measure: RD={risk_difference: .3f} and p0={baseline_prevalence: .3f} implies p1={treatment_prevalence: .3f}")
            rdiff = risk_difference
            rratio = treatment_prevalence / \
                baseline_prevalence
            oratio = treatment_prevalence * (1 - baseline_prevalence) / \
                (baseline_prevalence * (1 - treatment_prevalence))

        if risk_ratio is not None:
            if risk_ratio < 0:
                raise Exception("Risk ratio must be in [0, ∞).")
            if risk_difference is not None or odds_ratio is not None:
                raise Exception("Only one effect measure must be provided.")

            treatment_prevalence = risk_ratio * baseline_prevalence
            if treatment_prevalence > 1:
                raise Exception(
                    f"Invalid measure: RR={risk_ratio: .3f} and p0={baseline_prevalence: .3f} implies p1={treatment_prevalence: .3f}")
            rdiff = treatment_prevalence - baseline_prevalence
            rratio = risk_ratio
            oratio = treatment_prevalence * \
                (1-baseline_prevalence) / \
                (baseline_prevalence * (1-treatment_prevalence))

        if odds_ratio is not None:
            if odds_ratio < 0:
                raise Exception("Odds ratio must be in [0, ∞).")
            if risk_difference is not None or risk_ratio is not None:
                raise Exception("Only one effect measure must be provided.")

            treatment_odds = odds_ratio * \
                (baseline_prevalence / (1-baseline_prevalence))
            treatment_prevalence = treatment_odds / (1+treatment_odds)

            if treatment_prevalence > 1:
                raise Exception(
                    f"Invalid measure: OR={odds_ratio: .3f} and p0={baseline_prevalence: .3f} implies p1={treatment_prevalence: .3f}")

            rdiff = treatment_prevalence - baseline_prevalence
            rratio = treatment_prevalence / baseline_prevalence
            oratio = odds_ratio

        return {
            "RD": rdiff,
            "RR": rratio,
            "OR": oratio
        }
