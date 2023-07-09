import numpy as np
import pandas as pd
from epiverse.models.model_specification import ModelSpecification


class LogisticRegression(ModelSpecification):

    def __init__(self):
        pass

    def fit(self, **kwargs) -> ModelSpecification:
        pass

    def predict(self, exposure: np.array, **kwargs):
        pass
