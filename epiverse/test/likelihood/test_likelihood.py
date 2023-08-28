import unittest
from epiverse.likelihood.binomial_loglikelihood import BinomialLogLikelihood
from epiverse.data.prostatic import ProstaticData
from scipy.special import expit
import numpy as np

import statsmodels.formula.api as smf


class TestLikelihood(unittest.TestCase):

    def test_binomial(self):
        data = ProstaticData.retrieve_data()
        # codebook = ProstaticData.retrieve_codebook()

        data["Dead"] = data["Status"].transform(lambda x: int(x >= 1))
        data_subset = data[["Dead", "Stage", "AgeYrs", "Wt"]].dropna()

        bin_ll = BinomialLogLikelihood(
            outcomes=data_subset["Dead"].to_numpy(),
            data=data_subset[["Stage", "AgeYrs", "Wt"]].to_numpy())

        log_reg = smf.logit("Dead ~ Stage + AgeYrs + Wt",
                            data=data_subset).fit()

        self.assertAlmostEqual(
            log_reg.llf,
            bin_ll(log_reg.params.to_numpy())
        )
