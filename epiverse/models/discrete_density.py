from epiverse.models.density_model_specification import DensityModelSpecification
import numpy as np
import pandas as pd
from typing import Dict, List
from collections.abc import Iterable


class DiscreteDensity(DensityModelSpecification):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # Shortcut if provide dataset as Pandas Dataframe.
        if "dataset" in kwargs and isinstance(kwargs["dataset"], pd.DataFrame):
            self.data = kwargs["dataset"]
            self.names = list(self.data.columns)
            return

        # Otherwise, parse through provided data to construct data.
        data_length = None
        i = 1

        self.covariate_data = {}
        # Process args for covariate data
        for a in args:
            arg = a
            if isinstance(arg, pd.DataFrame):
                arg = a.to_numpy()
            elif not isinstance(arg, np.ndarray):
                continue

            if data_length is None:
                data_length = arg.shape[0]

            if arg.shape[0] != data_length:
                raise Exception(
                    f"Length of argument {arg} ({arg.shape[0]}) does not equal length of data ({data_length})")
            self.covariate_data["L" + str(i)] = arg
            i += 1

        # Process kwargs for covariate data
        for k, value in kwargs.items():
            v = value
            if isinstance(v, pd.Series):
                v = value.to_numpy()
            if isinstance(v, dict):
                v = pd.Series(value, name=k).to_numpy()

            if not isinstance(v, np.ndarray):
                continue
            if data_length is None:
                data_length = v.shape[0]

            if v.shape[0] != data_length:
                raise Exception(
                    f"Length of {k} ({v.shape[0]}) does not equal length of data ({data_length})")

            self.covariate_data[k] = v

        self.data = pd.DataFrame(data=self.covariate_data)
        self.names = list(self.data.columns)
        # Guess about whether this density is marginal or conditional.
        self.is_conditional = True
        self._is_fit = False
        self.conditioning_set = None

    def fit(self, event_variable: str | int, conditioning_set: str | int | List = None, conditioning_values: pd.DataFrame = None) -> DensityModelSpecification:
        # Error checking on event_variable
        if event_variable is None:
            raise Exception(
                "Must provide a variable to calculate the probability of.")

        if isinstance(event_variable, str):
            if event_variable not in self.names:
                raise Exception(
                    f"Event variable {event_variable} not found in list of names ({self.names})")

            self.event_variable = event_variable
        elif isinstance(event_variable, int):
            if event_variable < 0 or event_variable >= len(self.names):
                raise Exception(
                    f"Index {event_variable} is not present in list of names ({self.names})"
                )
            self.event_variable = self.names[event_variable]
        elif isinstance(event_variable, list) and len(event_variable) == 1:
            if isinstance(event_variable[0], str):
                self.event_variable = event_variable[0]
            elif isinstance(event_variable[0], int):
                self.event_variable = self.names[event_variable[0]]
            else:
                raise Exception(
                    f"If mistakenly passing a list, must be only with single object within, but was size ({len(event_variable)})")
        else:
            raise Exception(
                f"Event variable must be str or int, but was {type(event_variable)}")

        # Error checking on conditioning set:
        if conditioning_set is None:
            self.is_conditional = False
        elif isinstance(conditioning_set, list) and len(conditioning_set) == 0:
            self.is_conditional = False
        elif isinstance(conditioning_set, list):
            all_names_in_dataset = [
                name in self.names for name in conditioning_set]

            if not all(all_names_in_dataset):
                invalid_names = [i for (i, v) in zip(
                    conditioning_set, all_names_in_dataset) if not v]
                raise Exception(
                    f"Columns {invalid_names} are not present in the dataset.")
            self.conditioning_set = conditioning_set
            self.is_conditional = True
        elif isinstance(conditioning_set, str):
            if not conditioning_set in self.names:
                raise Exception(
                    f"Column {conditioning_set} not present in the columns {self.names}")

            self.conditioning_set = [conditioning_set]
            self.is_conditional = True
        elif isinstance(conditioning_set, int):
            if conditioning_set < 0 or conditioning_set >= len(self.names):
                raise Exception(
                    f"Provided conditioning_set index ({conditioning_set}) outside of range [0, {len(self.names)})")

            self.conditioning_set = [self.names[conditioning_set]]
            self.is_conditional = True
        else:
            raise Exception(
                f"conditioning_set must be a list, str, int, or None, but was {type(conditioning_set)}")

        # If marginal probability, just return self at this point.
        if not self.is_conditional:
            self._is_fit = True
            return self

        # Error checking on Conditioning Values, as not all values need to be calculated.
        # If no values are given, produce all possible combinations
        if conditioning_values is None:
            self.conditioning_values = self.data[self.conditioning_set].drop_duplicates(
            )
            self._is_fit = True
            return self

        elif not isinstance(conditioning_values, pd.DataFrame):
            if isinstance(conditioning_values, np.ndarray) or isinstance(conditioning_values, list):
                conditioning_values = pd.DataFrame(
                    data=conditioning_values, columns=self.conditioning_set)
            else:
                raise Exception(
                    f"Conditioning values must be Pandas dataframe, list of lists, or numpy array, but was {type(conditioning_values)}")

        if len(conditioning_values.columns) != len(self.conditioning_set):
            raise Exception(
                f"Conditioning set ({len(self.conditioning_set)}) and conditioning values ({len(conditioning_values.columns)}) different sizes.")
        elif not all(problem_columns := [c in conditioning_values.columns for c in self.conditioning_set]):
            columns_not_in_conditioning_set = [i for (i, v) in zip(
                conditioning_values.columns, problem_columns) if not v]
            raise Exception(
                f"Columns {columns_not_in_conditioning_set} not present in the conditioning set ({self.conditioning_set})")
        else:
            self.conditioning_values = conditioning_values.drop_duplicates()

        self._is_fit = True
        return self

    def predict(self, exposure: List):
        if not self._is_fit:
            raise Exception("Ensure model is fit before predicting values.")

        if not isinstance(exposure, Iterable):
            raise Exception(
                "Exposure must be an iterable object such as list or numpy array.")

        exposure_list = list(exposure)
        # The exposure parameter are the values to check.
        if not self.is_conditional:
            e = {(el, ): self._marginal_probability(el)
                 for el in exposure_list}
            self.probability_lookup = e
            return e

        e = [self._conditional_probability(el) for el in exposure_list]

        self.probability_lookup = {
            k: v for dict_in_list in e for k, v in dict_in_list.items()}

        return self.probability_lookup

    def _marginal_probability(self, exposure: int):

        exposure_matches = self.data[self.event_variable] == exposure

        return exposure_matches.mean()

    def _conditional_probability(self, exposure: int):

        # First, create masks for each set of conditioning values.
        mask = {}
        for index, row in self.conditioning_values.iterrows():
            mask_elements = self.data[self.conditioning_set] == row
            # Check if all values met
            mask_list = mask_elements.values.all(axis=1)
            mask_key = (exposure, *row.tolist())
            mask[mask_key] = mask_list.tolist()

        # Then, apply mask, check equality, and get mean.
        masked_mean = {
            k: (self.data[v][self.event_variable] == exposure).mean()
            for k, v, in mask.items()
        }

        return masked_mean

    def __str__(self):
        if self.is_conditional and self._is_fit:
            descr = f"Discrete density: Pr[{self.event_variable} | {self.conditioning_set}]"
        elif self._is_fit:
            descr = f"Discrete density: Pr[{self.event_variable}]"
        else:
            descr = f"Unfit Discrete density model"
        return descr
