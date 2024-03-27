from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Union, Callable


class FunctionModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, **kwargs) -> Callable:
        pass

    @abstractmethod
    def predict(self, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        pass

    @abstractmethod
    def predict_random(self, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        pass
