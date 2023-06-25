from typing import Callable, Tuple
from epiverse.models.model_specification import ModelSpecification
import numpy as np


# TODO: need method for iteratively reweighted least squares.
# Inputs: outcomes and inputs, optional starting weights matrix
# Have be recursive, with compiling?


class IterativelyReweightedLeastSquares(ModelSpecification):

    def __init__(self, p: int = 2):
        """Iteratively Reweighted Least Squares is a method for finding the optimal @beta
        solutions for a L-p norm problem. 

        Args:
            p (int, optional): The p-norm used in the problem; defaults to 2. 
        """

        self.p = p

    # For implementation details
    def fit(self, outcome: np.array, exposure: np.array, ε=1e-6) -> Tuple[np.array]:

        if outcome.shape[0] != exposure.shape[0]:
            raise Exception(
                f"Shapes of outcome vector ({outcome.shape}) and exposure vector ({exposure.shape}) not compatable.")

        i = 0
        difference = 1e6
        previous_beta = 1e6
        weights = np.eye(N=exposure.shape[0])

        while difference > ε:
            beta = np.linalg.inv(exposure.T @ weights @
                                 exposure) @ exposure.T @ weights @ outcome
            weights = np.diag((outcome - exposure @ beta)**(self.p-2))

            difference = np.abs(np.linalg.norm(
                previous_beta) - np.linalg.norm(beta))
            previous_beta = beta
            i += 1
        return beta, weights
