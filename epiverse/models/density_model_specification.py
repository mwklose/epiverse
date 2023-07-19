from abc import ABC, abstractmethod
from epiverse.models.model_specification import ModelSpecification
from typing import List


class DensityModelSpecification(ModelSpecification, ABC):

    def __init__(self):
        self._is_fit = False

    @abstractmethod
    def fit(self, event_variable, conditioning_set, conditioning_values):
        pass

    @abstractmethod
    def predict(self, exposure: List):
        pass
