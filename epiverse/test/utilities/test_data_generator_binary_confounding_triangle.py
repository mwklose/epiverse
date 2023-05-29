import unittest
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from epiverse.utilities.data_generation.data_generator_binary_confounding_triangle import DataGeneratorBinaryConfoundingTriangle


class TestDataGeneratorBinaryConfoundingTriangle(unittest.TestCase):

    def test_initialization(self):
        self.assertRaises(Exception, DataGeneratorBinaryConfoundingTriangle)
        self.assertRaises(
            Exception, DataGeneratorBinaryConfoundingTriangle, 100, None, 1, 1, 1)
        self.assertRaises(
            Exception, DataGeneratorBinaryConfoundingTriangle, 100, 1, None, 1, 1)
        self.assertRaises(
            Exception, DataGeneratorBinaryConfoundingTriangle, 100, 1, 1, None, 1)
        self.assertRaises(
            Exception, DataGeneratorBinaryConfoundingTriangle, 100, 1, 1, 1, -1)

    def test_generate_data(self):
        def trial(trt=0.5, cnf=0.25, out=0.3, oddsratio=2.0):
            dgb = DataGeneratorBinaryConfoundingTriangle(
                n=200000, treatment_prevalence=trt, confounder_prevalence=cnf, outcome_prevalence=out, odds_ratio=oddsratio)

            data = dgb.generate_data()

            prevalence = data.agg(np.mean).to_numpy()
            equal = np.isclose(prevalence, np.array(
                [cnf, trt, out]), rtol=2e-1)

            self.assertTrue(np.all(equal))
            odds = data.groupby(["W"]).apply(
                lambda x: (np.sum(x.Y * x.A) / np.sum((1 - x.Y) * x.A)) /
                (np.sum(x.Y * (1-x.A)) / np.sum((1 - x.Y) * (1-x.A)))
            )

            or_close = np.isclose(odds, oddsratio, rtol=1e-1)
            self.assertTrue(np.all(or_close))

        trial()

        trial(0.4, 0.4, 0.2, 1.5)
        trial(0.6, 0.2, 0.15, 3.0)

    def test_generate_by_counterfactual(self):

        def trial(y1=0.4, y0=0.2, treatment_prevalence=0.5, confounder_prevalence=0.25):

            dgb = DataGeneratorBinaryConfoundingTriangle(
                n=10000,
                treatment_prevalence=treatment_prevalence,
                confounder_prevalence=confounder_prevalence,
                y1=y1,
                y0=y0
            )

            data = dgb.generate_data()

            obs_y = data.groupby(["A"]).apply(
                lambda x: np.mean(x.Y)
            )

            counterfactuals_close = np.isclose(
                obs_y, np.array([y0, y1]), atol=5e-2)
            print(f"ObsY: {obs_y}\nTest: {counterfactuals_close}\n")
            self.assertTrue(np.all(counterfactuals_close))

        trial()
        trial(0.5, 0.3)
        trial(0.5, 0.3, 0.4, 0.2)
        trial(0.3, 0.15, 0.75, 0.5)
