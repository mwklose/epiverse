from epiverse.likelihood.likelihood import Likelihood
from inspect import signature


class ProfileLikelihood(Likelihood):
    def __init__(self, likelihood_contribution: Callable = None, loglikelihood_contribution: Callable = None, data: np.array = None):
        super().__init__(likelihood_contribution, loglikelihood_contribution, data)

        self.profile = {}

    def generate_profile(self, variable_str: str = None):
        likelihood_signature = signature(self.likelihood_contribution)
        loglikelihood_signature = signature(self.loglikelihood_contribution)

        if not variable_str in likelihood_signature and not variable_str in loglikelihood_signature:
            raise Exception(
                f"Variable {variable_str} not found in {likelihood_signature} or {loglikelihood_signature}")

    def confidence_interval():
        pass
