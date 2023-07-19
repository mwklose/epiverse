from abc import ABC, abstractmethod
from epiverse.models.model_specification import ModelSpecification


class OutcomeModelSpecification(ModelSpecification, ABC):

    def __init__(self):
        self._is_fit = False

    @abstractmethod
    def fit(self, outcome, exposure):
        pass

    @abstractmethod
    def predict(self, observations):
        pass
