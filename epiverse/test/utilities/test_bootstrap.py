import unittest
import pandas as pd
from typing import List
from epiverse.utilities.variance.bootstrap import Bootstrap
from epiverse.data.prostatic import ProstaticData


def mean_prostatic(data: pd.DataFrame, columns: List):
    return data[columns].mean(axis=0)


class TestBootstrap(unittest.TestCase):

    def test_bootstrap(self):
        data = ProstaticData.retrieve_data()

        bs = Bootstrap(
            data=data, bootstrap_function=mean_prostatic, number_of_iterations=1000, columns=["SBP", "DBP"])

        print(bs.results.std(axis=0))
