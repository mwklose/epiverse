from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Union


class EffectModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def result(self) -> Union[np.ndarray, pd.Series]:
        """Generates an R-like table with the coefficient names, values, standard errors, all in one table

        Returns:
            Union[np.ndarray, pd.Series]: a table-like structure with the results form the model fit. 
        """
        pass

    @abstractmethod
    def vcov(self, **kwargs) -> np.ndarray:
        """Accesses the variance-covariance matrix for the model, if applicable. 

        Returns:
            np.ndarray: the variance matrix. 
        """
        pass

    @abstractmethod
    def params(self, **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Returns relevant coefficients

        Returns:
            Union[np.ndarray, pd.DataFrame]: the coefficients of the model
        """
        pass
