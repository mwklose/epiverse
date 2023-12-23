from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Union


class EffectModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def result(self) -> Union[np.ndarray, pd.Series]:
        pass

    @abstractmethod
    def vcov(self, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def params(self, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        pass
