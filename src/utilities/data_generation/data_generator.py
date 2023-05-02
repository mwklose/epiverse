
from abc import ABC, abstractmethod
from typing import Dict
# Parent class for methods to generate for
# Abstract Python class needed.


class DataGenerator(ABC):

    @abstractmethod
    def generate_data(self):
        pass
