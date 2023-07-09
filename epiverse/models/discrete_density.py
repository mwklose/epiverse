from epiverse.models.model_specification import ModelSpecification
import numpy as np
import pandas as pd
from typing import Dict, List


class DiscreteDensity(ModelSpecification):

    def __init__(self, *args, **kwargs):

        data_length = None
        i = 1

        self.covariate_data = {}
        # Process args for covariate data
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                arg = arg.to_numpy()
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
        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                continue
            if data_length is None:
                data_length = v.shape[0]

            if v.shape[0] != data_length:
                raise Exception(
                    f"Length of {k} ({v.shape[0]}) does not equal length of data ({data_length})")

            self.covariate_data[k] = v

        self.args = args
        self.kwargs = kwargs

    def fit(self, **kwargs) -> ModelSpecification:

        self.kwargs.update(kwargs)

        self.names = list(self.covariate_data.keys())
        args_to_stack = [np.array(list(a))
                         for a in self.covariate_data.values()]
        self.observed_array = np.vstack([*args_to_stack]).T

        return self

    def predict(self, exposure: np.array, **kwargs):
        """Predict the marginal probability (if no conditioning set is provided) or 
        conditional probability (if 'conditioning_set' is provided) for a given exposure level. 

        Args:
            exposure (np.array): a 1D numpy array containing the exposure history to control for
        """
        self.kwargs.update(kwargs)

        # Check for event variable, which is needed
        if "event_variable" not in self.kwargs:
            raise Exception(
                "Must provide a variable to calculate the probability of.")

        if isinstance(self.kwargs["event_variable"], str) and self.kwargs["event_variable"] not in self.names:
            raise Exception(
                f"Event variable {self.kwargs['event_variable']} not found in list of names ({self.names})")
        elif isinstance(self.kwargs["event_variable"], int) and (self.kwargs["event_variable"] < 0 or self.kwargs["event_variable"] >= len(self.names)):
            raise Exception(
                f"Index {self.kwargs['event_variable']} is not present in list of names ({self.names})"
            )

        event_variable = self.kwargs['event_variable']

        # Check if conditional
        if not "conditioning_set" in self.kwargs:
            # If want marginal, extract the variable to marginalize over
            return self._marginal_probability(exposure, event_variable)

        # Otherwise, focus on conditional probability
        if not isinstance(self.kwargs["conditioning_set"], list) and \
                not isinstance(self.kwargs["conditioning_set"], int) and \
                not isinstance(self.kwargs["conditioning_set"], str):
            raise Exception(
                f"Conditioning set not list, int, or str, was {type(self.kwargs['conditioning_set'])}")

        conditioning_set = self.kwargs["conditioning_set"]

        if "conditioning_values" not in self.kwargs:
            raise Exception(
                f"Provided conditioning set, but not values to condition on.")
        elif not isinstance(self.kwargs["conditioning_values"], np.ndarray):
            raise Exception(
                f"Conditioning values should be provided as an NxP \
                    array for the P items in conditioning set, was {type(self.kwargs['conditioning_values'])}"
            )

        conditioning_values = self.kwargs["conditioning_values"]

        return self._conditional_probability(exposure, event_variable, conditioning_set, conditioning_values)

    def get_names(self):
        """Returns the names and order of covariate data in the dataset.

        Returns:
            List: a list of all the names supplied in the original call to the function.
        """
        return self.names

    def _marginal_probability(self, exposure: np.array, marginal_variable: int | str):

        if isinstance(marginal_variable, str):
            if marginal_variable not in self.names:
                raise Exception(
                    f"Variable {marginal_variable} not in list of names ({self.names}).")

            exposure_column = self.names.index(marginal_variable)
        elif isinstance(marginal_variable, int):
            if marginal_variable < 0 or marginal_variable >= len(self.names):
                raise Exception(
                    f"Marginal variable index {marginal_variable} is not in the range [0, {len(self.names)})")

            exposure_column = marginal_variable
        else:
            raise Exception(
                f"Marginal variable should be an int or a string, but was {type(marginal_variable)}")

        observations_to_check = self.observed_array[:, exposure_column]

        x = np.mean(observations_to_check
                    == exposure[:, None], axis=1)
        return x

    def _conditional_probability(self, exposure: np.array,
                                 event_variable: int | str,
                                 conditioning_set: int | str | List,
                                 conditioning_values: np.array) -> np.array:
        """Calculates the conditional probability of the event variable equaling the supplied exposure, 
        conditional on the provided set and values that set can take. 

        For testing P exposures (such as P=2 when `exposure=np.array([1,0])`) against 
        N different conditioning sets, the resulting numpy array will be NxP. 

        Args:
            exposure (np.array): a numpy array corresponding to the levels of exposure wanted. 
            event_variable (int | str): a variable telling which column of observed data to use as the event
            conditioning_set (int | str | List): a set of variables telling which columns of observed data to use as the conditioning set.
                Cannot contain the event variable.
            conditioning_values (np.array): a numpy array corresponding to the levels of that the conditioning set will take on

        Raises:
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 
            Exception: Invalid input checking. 

        Returns:
            np.array: an np.array that has the same number of rows as conditioning_values, 
                the same number of columns as exposure, 
                where each entry corresponds to the conditional probability. 
        """
        # Check validity of event variable
        if isinstance(event_variable, str):
            if event_variable not in self.names:
                raise Exception(
                    f"Variable {event_variable} not in list of names ({self.names}).")

            exposure_column = self.names.index(event_variable)
        elif isinstance(event_variable, int):
            if event_variable < 0 or event_variable >= len(self.names):
                raise Exception(
                    f"Marginal variable index {event_variable} is not in the range [0, {len(self.names)})")

            exposure_column = event_variable
        else:
            raise Exception(
                f"Event variable should be an int or a string, but was {type(event_variable)}")

        # Check validity of conditioning set
        if isinstance(conditioning_set, int):
            if conditioning_set < 0 or conditioning_set >= len(self.names):
                raise Exception(
                    f"Conditioning set index must be valid, was {conditioning_set}")

            if event_variable == conditioning_set:
                raise Exception(
                    f"Event variable ({event_variable}) cannot also be a conditioning variable.")
            conditioning_columns = [conditioning_set]
        elif isinstance(conditioning_set, str):
            if conditioning_set not in self.names:
                raise Exception(
                    f"Conditioning variable {conditioning_set} not in list of names ({self.names}).")

            if event_variable == (conditioning_set_index := self.names.index(conditioning_set)):
                raise Exception(
                    f"Event variable ({event_variable}) cannot also be a conditioning variable.")
            conditioning_columns = [conditioning_set_index]

        elif isinstance(conditioning_set, list):
            conditioning_columns_check_str = [
                c in self.names for c in conditioning_set]
            conditioning_columns_check_int = [c in range(
                len(self.names)) for c in conditioning_set]
            conditioning_columns_check = [a or b for a, b in zip(
                conditioning_columns_check_str, conditioning_columns_check_int)]

            if not all(conditioning_columns_check):
                raise Exception(
                    f"Not all values in conditioning set are valid: {conditioning_set}")

            if len(conditioning_columns_check) != len(conditioning_set):
                raise Exception(
                    f"Duplicates foound within conditioning columns, parsed as {conditioning_columns_check}")
            if event_variable in conditioning_set:
                raise Exception(
                    f"Event variable ({event_variable}) cannot also be a conditioning variable ({conditioning_columns_check}).")

            conditioning_columns = [
                self.names.index(c) for c in conditioning_set
            ]

        else:
            raise Exception(
                f"Invalid conditioning set type, must be str, int, or list but was {type(conditioning_set)}")
        # For conditioning list, convert all to integers first.

        observations_from_conditioning_set = self.observed_array[:,
                                                                 conditioning_columns]
        # Conditional filter is a 3D array
        conditional_filter = (observations_from_conditioning_set ==
                              conditioning_values[:, None])

        # Handle case of single conditioning set
        if len(conditional_filter.shape) <= 2:
            conditional_mean = np.mean(
                self.observed_array[conditional_filter.flatten(), exposure_column] == exposure[:, None])
            return conditional_mean

        # Collapse conditional filter by 1 dimension through using "all"
        conditional_filter_valid_rows = np.all(
            conditional_filter, axis=-1)

        def conditional_mean_of_array(x):
            """From an input which filters the rows that meet conditioning criteria, 
            return the mean of the exposure column

            Args:
                x (_type_): A boolean masking vector corresponding to observations which match conditioning set. 

            Returns:
                _type_: the probability of event variable equaling the supplied exposure, across all the conditioning set combinations
            """
            observed_array_subset = self.observed_array[x, exposure_column]
            conditional_mean = observed_array_subset == exposure[:, None]

            row_conditional_means = np.mean(conditional_mean, axis=1)
            return row_conditional_means

        means = np.apply_along_axis(
            conditional_mean_of_array, axis=-1, arr=conditional_filter_valid_rows)

        return means
