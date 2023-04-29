
from abc import ABC, abstractmethod
from typing import Dict
# Parent class for methods to generate for
# Abstract Python class needed.


class DataGenerator(ABC):

    @abstractmethod
    def generate_data_by_prevalence(self):
        pass

    @abstractmethod
    def generate_data_by_effect_measure(self):
        pass

    @abstractmethod
    def check_effect_measures(self, baseline_prevalence: float = None, risk_difference: float = None, risk_ratio: float = None, odds_ratio: float = None) -> Dict:
        pass
