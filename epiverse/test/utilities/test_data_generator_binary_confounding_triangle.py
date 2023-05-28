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
