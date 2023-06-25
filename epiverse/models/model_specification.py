from abc import ABC, abstractmethod


class ModelSpecification(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, outcome, exposure, initial_weights, ε=1e-6):
        pass
