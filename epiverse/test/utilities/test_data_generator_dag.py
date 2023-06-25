import unittest
import numpy as np
import pandas as pd

from epiverse.utilities.dag.dag import DAG
from epiverse.utilities.data_generation.data_generator_dag import DataGeneratorDAG


class TestDataGeneratorDAG(unittest.TestCase):

    def test_generate_data(self):
        dag = DAG.binary_confounding_triangle(0.5, 0.25, 0.15, 0.3, 100000)

        dgd = DataGeneratorDAG(dag)
        data = dgd.generate_data()

        print(data.head())
        print(data.mean())
        print(data.groupby(["A"]).mean())

        mbias_dag = DAG.binary_m_bias(0.5, 0.4, 0.15, 0.3, 0.25, 0.25, 100000)
        mbd = DataGeneratorDAG(mbias_dag)
        data_m = mbd.generate_data()
        print(data_m.head())
        print(data_m.mean())
        print(data.groupby(["A"]).mean())
