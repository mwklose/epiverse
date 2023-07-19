from epiverse.models.density_model_specification import DensityModelSpecification
from epiverse.models.outcome_model_specification import OutcomeModelSpecification
from epiverse.models.model_specification import ModelSpecification
import numpy as np
import pandas as pd
from typing import List

class GComputation(ModelSpecification):
    
    def __init__(self, outcome_model: OutcomeModelSpecification, density_models: List[DensityModelSpecification], *args, **kwargs):
        if not isinstance(density_models, list) or len(density_models) < 2:
            raise Exception(f"At least 2 density models must be supplied as a list.")
            
        self.outcome_model = outcome_model
        self.density_models = density_models
        self.n_models = len(self.density_models)
        self.args = args
        self.kwargs = kwargs
        self.labels = None
        self._is_fit = False
    
    
    def fit(self, data: pd.DataFrame | np.ndarray, outcome_column: str, treatment_columns: List, 
            covariate_columns: List, covariate_conditioning_sets: List, **kwargs) -> ModelSpecification:
        
        # Step 0: get all unique covariate values to check. Prevents unnecessary computation.
        if isinstance(data, np.ndarray):
            if "labels" in kwargs: 
                self.labels = fit_kwargs["labels"]
            else: 
                raise Exception("Provide labels for each column in data.")
            self.data = pd.DataFrame(data, columns=self.labels)
            
        elif isinstance(data, pd.DataFrame): 
            self.data = data
            self.labels = list(data)
        else:
            raise Exception(f"Data must be passed as numpy array or pandas DataFrame, but was {type(data)}")
        
        # Argument checking to see if all is there
        if not isinstance(outcome_column, str): 
            raise Exception("Output column must be provided as a string")
        
        if outcome_column not in self.labels: 
            raise Exception(f"Provided outcome column ({outcome_column}) not in list of labels ({self.labels})")
        self.outcome_column = outcome_column
        self.outcome_index = self.labels.index(outcome_column)
        
        
        if not isinstance(treatment_columns, list): 
            raise Exception("Treatment columns must be provided as a list.")
        
        if not all([tc in self.labels for tc in treatment_columns]):
            raise Exception("Not all provided treatment columns are labeled in the dataset.")
        self.treatment_columns = treatment_columns
        self.treatment_indices = [self.labels.index(tc) for tc in treatment_columns if tc in self.labels]
        
        
        if not isinstance(covariate_columns, list): 
            raise Exception("Covariate columns must be provided as a list or lists, even if only one element.")
        if not isinstance(covariate_columns[0], list):
            raise Exception("Covariate columns must be provided as a list or lists, even if only one element.")
        self.covariate_columns = covariate_columns
        self.covariate_columns_index = [self.labels.index(*cci) for cci in covariate_columns]
        
        if not isinstance(covariate_conditioning_sets, list):
            raise Exception("Covariate conditioning set, even if empty, must be supplied and same length as density columns.")
        if len(covariate_conditioning_sets) != len(covariate_columns):
            raise Exception(f"Covariate conditioning set ({len(covariate_conditioning_sets)}) does not have same number of columns as covariate columns ({len(covariate_columns)})")

        if not all([cs in self.labels for conditioning_set in covariate_conditioning_sets for cs in conditioning_set ]):
            raise Exception("Covariate conditioning sets contain elements which are not labeled in dataset.")
        self.covariate_conditioning_sets = covariate_conditioning_sets
        self.covariate_indices = [[self.labels.index(cs) for cs in conditioning_set] for conditioning_set in covariate_conditioning_sets]
        
        # Step 1: Fit the Outcome Model
        self.outcome_model.fit(outcome=self.data[self.outcome_column].to_numpy(), 
                               exposure=self.data[self.covariate_conditioning_sets[-1]].to_numpy())
        
        # Step 2: Fit each of the varying density models
        for i, d_model in enumerate(self.density_models):
            d_model.fit(
                event_variable=covariate_columns[i], 
                conditioning_set=covariate_conditioning_sets[i-1] if i > 0 else []
            )
            print(f"{i}: {d_model}")

        
        self._is_fit = True
        return self
        
            
    # The predict procedure takes in a treatment history (eventually change to treatment history function?)
    # - Upon prediction, the density models must predict under the provided exposure, 
    # - The densities are accumulated through product
    # - The densities are multiplied by the expected outcome
    # - Each piece's accumulation is then added together over all possibilities.
    def predict(self, exposure: List, **kwargs):
        
        if not isinstance(exposure, list) or len(exposure) != len(self.treatment_indices):
            raise Exception("Exposure must be a list with the same length as the number of treatment columns")
        
        # Deep copy needed because otherwise, we would overwrite the original data.
        gcomp = self.data.copy(deep=True)
        gcomp_labels = self.labels.copy()
        
        for i, e in enumerate(exposure): 
            gcomp[self.treatment_columns[i]] = e
        
        
        # Step 1: predict the outcome model under augmented data
        outcome_subset = gcomp[self.covariate_conditioning_sets[-1]].drop_duplicates()
        outcome_subset_labels = [gcomp_labels[i] for i in self.covariate_indices[-1]]
        
        gcomp["predicted_outcomes"] = self.outcome_model.predict(gcomp[self.covariate_conditioning_sets[-1]].to_numpy())
        # Step 2: for each of the density models, get a prediction as well.
        gcomp[f"acc_density"] = 1

        for i, d_model in enumerate(self.density_models):
            # Make predictions for having either covariate under all histories
            # Note - this is not efficient, and there is a more pythonic way of doing this.
            predicted_density = d_model.predict(exposure=[0,1])
            
            # Retrieve columns that are relevant
            density_model_outcome = self.covariate_columns[i]
            density_model_covariate = self.covariate_conditioning_sets[i-1] if i > 0 else []
            density_model_form = [*density_model_outcome, *density_model_covariate]
            gcomp_column_names = list(gcomp.columns)
            density_model_indices = [gcomp_column_names.index(i) for i in density_model_form]
            
            def access_densities(row):
                access_tuple = tuple(row[density_model_indices])
                val = predicted_density[access_tuple]
                return val
        
            gcomp[f"density_{i}"] = np.apply_along_axis(access_densities, 1, gcomp)
            gcomp["acc_density"] *= gcomp[f"density_{i}"]

        # Step 3: accumulate the results
        gcomp["gcomp_contribution"] = gcomp["predicted_outcomes"] * gcomp["acc_density"]
        
        # Funny business here: easier to calculate for all, then find the unique histories than subsetting first.
        l_history_columns = [l[0] for l in self.covariate_columns]
        unique_histories = gcomp[l_history_columns].drop_duplicates().index
        
        return gcomp.iloc[unique_histories]["gcomp_contribution"].sum()
        
        
        
        