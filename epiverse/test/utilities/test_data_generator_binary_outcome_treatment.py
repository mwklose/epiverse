import unittest
from epiverse.utilities.data_generation.data_generator_binary_outcome import DataGeneratorBinaryOutcomeTreatment
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
        with self.assertRaises(Exception) as test_ar:
            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, risk_ratio=-1)
            self.assertEqual(str(test_ar.exception),
                             "Odds ratio must be in [0, ∞).")

            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, risk_ratio=1.1, risk_difference=0.4)
            self.assertEqual(str(test_ar.exception),
                             "Only one effect measure must be provided.")

            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, risk_ratio=3.0)
            self.assertEqual(str(test_ar.exception),
                             "Invalid measure: RR=3.000 and p0=0.500 implies p1=1.500")

        results = DataGeneratorBinaryOutcomeTreatment(
            n=10000, baseline_prevalence=0.5, risk_ratio=1)

        risks = results.data.groupby(
            ["Exposure"]).mean()
        risk_ratio = risks.iloc[1, 0] / risks.iloc[0, 0]

        self.assertAlmostEqual(
            risk_ratio, 1, msg="Risk Ratio Null input; stochastic", delta=0.1)

        results2 = DataGeneratorBinaryOutcomeTreatment(
            n=15000, baseline_prevalence=0.5, risk_ratio=1.5)

        risks2 = results2.data.groupby(
            ["Exposure"]).mean()
        risk_ratio2 = risks2.iloc[1, 0] / risks2.iloc[0, 0]
        self.assertAlmostEqual(risk_ratio2, 1.5, delta=0.15,
                               msg="Risk Ratio 1.5 input; stochastic")

    def test_pass_RD(self):
        with self.assertRaises(Exception) as test_ar:
            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, risk_difference=-1.1)
            self.assertEqual(str(test_ar.exception),
                             "Risk difference must be in [-1, 1].")
            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, risk_difference=1.4)
            self.assertEqual(str(test_ar.exception),
                             "Risk difference must be in [-1, 1].")

            DataGeneratorBinaryOutcomeTreatment(
                baseline_prevalence=0.5, risk_difference=0.7)
            self.assertEqual(str(test_ar.exception),
                             "Invalid measure: RD=0.700 and p0=0.500 implies p1=1.200")

        results = DataGeneratorBinaryOutcomeTreatment(
            n=10000, baseline_prevalence=0.5, risk_difference=0)

        risks = results.data.groupby(
            ["Exposure"]).mean()
        risk_difference = risks.iloc[1, 0] - risks.iloc[0, 0]

        self.assertAlmostEqual(
            risk_difference, 0, msg="Risk Ratio Null input; stochastic", delta=0.1)

        results2 = DataGeneratorBinaryOutcomeTreatment(
            n=10000, baseline_prevalence=0.1, risk_difference=0.2)

        risks2 = results2.data.groupby(
            ["Exposure"]).mean()
        risk_difference2 = risks2.iloc[1, 0] - risks2.iloc[0, 0]
        self.assertAlmostEqual(risk_difference2, 0.2, delta=0.1,
                               msg="Risk Difference 0.2 input; stochastic")
