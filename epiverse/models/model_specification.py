from abc import ABC, abstractmethod


class ModelSpecification(ABC):

    def __init__(self):
        self._is_fit = False

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, exposure, **kwargs):
        pass
