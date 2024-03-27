from epiverse.utilities.data_generation import DataGeneratorWen, DataGeneratorWenSimulation
import numpy as np
from icecream import ic


def test_wen():
    default_wen = DataGeneratorWen()
    default_wen_sim = DataGeneratorWenSimulation()

    assert not default_wen.data.empty
    assert not default_wen_sim.data.empty
