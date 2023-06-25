import unittest
from epiverse.likelihood.likelihood import Likelihood
from epiverse.data.prostatic import ProstaticData
from scipy.special import expit
import numpy as np


class TestLikelihood(unittest.TestCase):

    def test_binomial(self):
        data = ProstaticData.retrieve_data()
        codebook = ProstaticData.retrieve_codebook()

        data["Dead"] = data["Status"].transform(lambda x: x >= 1)

        def ll(parameters, data):
            p1 = expit(parameters[0] + parameters[1] * data["BM"])

            b_ll = data["Dead"] * np.log(p1) + \
                (1 - data["Dead"]) * np.log(1 - p1)

            total_ll = np.sum(b_ll)
            return total_ll

        bin_ll = Likelihood(
            loglikelihood_contribution=ll,
            data=data[["BM", "Dead"]]
        )

        print(bin_ll.full_logeval(np.array([0, 0])))

        print(bin_ll.maximize(initial_values=[0, 0]))
