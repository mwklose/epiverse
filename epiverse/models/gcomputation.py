from epiverse.models.model_specification import ModelSpecification
import numpy as np
from typing import List

class GComputation(ModelSpecification):
    
    def __init__(self, outcome_model: ModelSpecification, density_models: List[ModelSpecification], *args, **kwargs):
        if not isinstance(density_models, list) or len(density_models) < 2:
            raise Exception(f"At least 2 density models must be supplied as a list.")
            
        
        self.outcome_model = outcome_model
        self.density_models = density_models
        self.n_models = len(self.density_models)
        self.args = args
        self.kwargs = kwargs
        
        
    def fit(self, **kwargs) -> ModelSpecification:
        fit_kwargs = {k: v for k,v in self.kwargs.items()}
        fit_kwargs.update(kwargs)
        
        print(fit_kwargs)
        if "outcome_column" not in fit_kwargs: 
            raise Exception("Output column must be provided as a string or int.")
        
        if "treatment_columns" not in fit_kwargs or not isinstance(fit_kwargs["treatment_columns"], list): 
            raise Exception("Treatment columns must be provided as a list.")
        
        if "covariate_columns" not in fit_kwargs or not isinstance(fit_kwargs["covariate_columns"], list): 
            raise Exception("Covariate columns must be provided as a list or lists, even if only one element.")
        if not isinstance(fit_kwargs["covariate_columns"][0], list):
            raise Exception("Covariate columns must be provided as a list or lists, even if only one element.")
        
        
        if "covariate_conditioning_sets" not in fit_kwargs:
            raise Exception("Covariate conditioning set, even if empty, must be supplied and same length as outcome columns.")
        
        
        
        
        # Items needed: 
        # 1. Full conditioning set for exposure model
        # 2. Conditioning set for covariate history
        
        
        # Fit the conditional outcome models
        # Only at final time point for G-Computation
        
        
        # Fit the conditional density models
        
        # accumulate Conditional density models
        
        # weighted sum over conditional outcome models is result
        
        
        if "data" not in fit_kwargs:
            raise Exception("Must specific a dataset using data=<dataset>")
        elif not isinstance(fit_kwargs["data"], np.ndarray):
            raise Exception(f"Data must be a Numpy Array, was: {type(fit_kwargs['data'])}")
        
        
        outcomes = [model.fit(self.kwargs) for model in self.outcome_model]
        density = [jp.fit(self.kwargs) for jp in self.density_model]
            

    def predict(self, exposure):
        pass
        
        