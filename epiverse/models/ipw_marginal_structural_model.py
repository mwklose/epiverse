from epiverse.models.density_model_specification import DensityModelSpecification
from epiverse.models.outcome_model_specification import OutcomeModelSpecification
from epiverse.models.model_specification import ModelSpecification
import numpy as np
import pandas as pd
from typing import List, Callable


class IPWMarginalStructuralModel(ModelSpecification):

    def __init__(self, density_models: List, stablized_constructor: Callable = None, *args, **kwargs):
        if not isinstance(density_models, list):
            raise Exception(f"Density models must be supplied as a list.")

        self.density_models = density_models
        self.n_models = len(self.density_models)

        self._is_stabilized = stablized_constructor is not None
        self.stabilized_constructor = stablized_constructor

        self.kwargs = kwargs
        self.labels = None
        self._is_fit = False

    def fit(self, data: pd.DataFrame | np.ndarray, outcome_column: str, treatment_columns: List,
            covariate_columns: List):

        # Step 0: get all unique covariate values to check. Prevents unnecessary computation.
        self.check_np_pd(data, **self.kwargs)
        self.check_outcome_column(outcome_column)
        self.check_treatment_columns(treatment_columns)
        self.check_covariate_columns(covariate_columns)

        # Step 1: fit each of the density models
        for i, d_model in enumerate(self.density_models):
            d_model.fit(
                event_variable=self.treatment_columns[i],
                conditioning_set=[
                    *self.treatment_columns[0:i], *self.covariate_columns[i]]
            )

        # Step 2: if there is stabilization, then make those models as well.
        if self._is_stabilized:
            self.stabilizing_models = []
            for i, treatment_column in enumerate(self.treatment_columns):
                treatment_history = self.treatment_columns[0:i+1]
                data_as_dict = self.data[treatment_history].to_dict()

                print(f"{i}: {treatment_column}")

                stabilizing_model = self.stabilized_constructor(
                    **data_as_dict).fit(
                        event_variable=treatment_column,
                        conditioning_set=self.treatment_columns[0:i]
                )

                print(stabilizing_model.data)

                self.stabilizing_models.append(
                    stabilizing_model
                )

        self._is_fit = True
        return self

    def predict(self, exposure: List):

        if not isinstance(exposure, list) or len(exposure) != len(self.treatment_indices):
            raise Exception(
                "Exposure must be a list with the same length as the number of treatment columns")

        ipw = self.data.copy(deep=True)
        ipw_labels = self.labels.copy()

        for i, e in enumerate(exposure):
            ipw[self.treatment_columns[i] + "_new"] = e

        ipw[f"acc_density"] = 1

        for i, d_model in enumerate(self.density_models):
            # Make predictions for having either covariate under all histories
            # Note - this is not efficient, and there is a more pythonic way of doing this.
            predicted_density = d_model.predict(exposure=[0, 1])

            print(predicted_density)

        #     # Retrieve columns that are relevant
        #     density_model_outcome = self.covariate_columns[i]

        #     density_model_form = [
        #         *density_model_outcome, *density_model_covariate]
        #     gcomp_column_names = list(gcomp.columns)
        #     density_model_indices = [
        #         gcomp_column_names.index(i) for i in density_model_form]

        #     def access_densities(row):
        #         access_tuple = tuple(row[density_model_indices])
        #         val = predicted_density[access_tuple]
        #         return val

        #     gcomp[f"density_{i}"] = np.apply_along_axis(
        #         access_densities, 1, gcomp)
        #     gcomp["acc_density"] *= gcomp[f"density_{i}"]

        # # Step 3: accumulate the results
        # gcomp["gcomp_contribution"] = gcomp["predicted_outcomes"] * \
        #     gcomp["acc_density"]

        # # Funny business here: easier to calculate for all, then find the unique histories than subsetting first.
        # l_history_columns = [l[0] for l in self.covariate_columns]
        # unique_histories = gcomp[l_history_columns].drop_duplicates().index

        # return gcomp.iloc[unique_histories]["gcomp_contribution"].sum()
