from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Callable


@DeprecationWarning
class ModelSpecification(ABC):

    def __init__(self):
        self._is_fit = False

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def estimate(self, **kwargs):
        pass

    @abstractmethod
    def vcov(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    def check_np_pd(self, dataset: pd.DataFrame, **kwargs):
        if isinstance(dataset, np.ndarray):
            if "labels" in kwargs:
                self.labels = self.kwargs["labels"]
            else:
                raise Exception("Provide labels for each column in data.")
            self.data = pd.DataFrame(dataset, columns=self.labels)

        elif isinstance(dataset, pd.DataFrame):
            self.data = dataset
            self.labels = list(dataset)
        else:
            raise Exception(
                f"Data must be passed as numpy array or pandas DataFrame, but was {type(dataset)}")

    def check_outcome_column(self, outcome_column: str):

        if not isinstance(outcome_column, str):
            raise Exception("Output column must be provided as a string")

        if not self.labels:
            raise Exception("ModelSpecification lacks labels currently.")
        if outcome_column not in self.labels:
            raise Exception(
                f"Provided outcome column ({outcome_column}) not in list of labels ({self.labels})")
        self.outcome_column = outcome_column
        self.outcome_index = self.labels.index(outcome_column)

    def check_treatment_columns(self, treatment_columns: List):
        if not isinstance(treatment_columns, list):
            raise Exception("Treatment columns must be provided as a list.")

        if not all([tc in self.labels for tc in treatment_columns]):
            raise Exception(
                "Not all provided treatment columns are labeled in the dataset.")
        self.treatment_columns = treatment_columns
        self.treatment_indices = [self.labels.index(
            tc) for tc in treatment_columns if tc in self.labels]

    def check_covariate_columns(self, covariate_columns: List):
        if not isinstance(covariate_columns, list):
            raise Exception(
                "Covariate columns must be provided as a list or lists, even if only one element.")
        if not isinstance(covariate_columns[0], list):
            raise Exception(
                "Covariate columns must be provided as a list or lists, even if only one element.")
        self.covariate_columns = covariate_columns
        self.covariate_columns_index = [
            self.labels.index(*cci) for cci in covariate_columns]
