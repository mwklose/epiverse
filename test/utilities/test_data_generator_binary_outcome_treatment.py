import unittest
from src.utilities.data_generation.data_generator_binary_outcome import DataGeneratorBinaryOutcomeTreatment
import numpy as np


class TestDataGeneratorBinaryOutcomeTreatment(unittest.TestCase):

    def test_no_args(self):
        with self.assertRaises(Exception) as test_ar:
            DataGeneratorBinaryOutcomeTreatment()
            self.assertEqual(
                str(test_ar.exception), "Baseline prevalence for effect measure must not be None.")

        with self.assertRaises(Exception) as test_ar:
            DataGeneratorBinaryOutcomeTreatment(baseline_prevalence=0.5)
            self.assertEqual(str(test_ar.exception),
                             "One non-None effect measure must be provided.")

    def test_pass_OR(self):
        with self.assertRaises(Exception) as test_ar:
            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, odds_ratio=-1)
            self.assertEqual(str(test_ar.exception),
                             "Odds ratio must be in [0, ∞).")

            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, odds_ratio=1.1, risk_difference=0.4)
            self.assertEqual(str(test_ar.exception),
                             "Only one effect measure must be provided.")

        results = DataGeneratorBinaryOutcomeTreatment(
            n=10000, baseline_prevalence=0.5, odds_ratio=1)

        odds = results.data.groupby(
            ["Exposure"]).mean().apply(lambda x: x / (1-x))
        odds_ratio = odds.iloc[1, 0] / odds.iloc[0, 0]

        self.assertAlmostEqual(
            odds_ratio, 1, msg="Odds Ratio Null input; stochastic", delta=0.1)

        results2 = DataGeneratorBinaryOutcomeTreatment(
            n=10000, baseline_prevalence=0.5, odds_ratio=1.5)

        odds2 = results2.data.groupby(
            ["Exposure"]).mean().apply(lambda x: x / (1-x))
        odds_ratio2 = odds2.iloc[1, 0] / odds2.iloc[0, 0]
        self.assertAlmostEqual(odds_ratio2, 1.5, delta=0.1,
                               msg="Odds Ratio 1.5 input; stochastic")

    def test_pass_RR(self):
        self.assertTrue(True)

    def test_pass_RD(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
