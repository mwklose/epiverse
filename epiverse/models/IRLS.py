from typing import Callable, Tuple
from epiverse.models import EffectModel
import numpy as np
from dataclasses import dataclass


@dataclass
class IterativelyReweightedLeastSquares(EffectModel):
    outcome: np.array
    exposure: np.array
    eps: float = 1e-6
    p: int = 2

    def __post_init__(self):
        if self.outcome.shape[0] != self.exposure.shape[0]:
            raise Exception(
                f"Shapes of outcome vector ({self.outcome.shape}) and exposure vector ({self.exposure.shape}) not compatable.")

        self.beta, self.weights = self._fit_procedure()
        pass

    # TODO: fix this to align with model specification better.
    # For implementation details
    def _fit_procedure(self) -> Tuple[np.array]:
        i = 0
        difference = 1e6
        previous_beta = 1e6
        weights = np.eye(N=self.exposure.shape[0])

        while difference > self.eps:
            ew = self.exposure.T @ weights
            beta = np.linalg.inv(ew @ self.exposure) @ ew @ self.outcome
            weights = np.diag(
                (self.outcome - self.exposure @ beta)**(self.p-2))

            difference = np.abs(np.linalg.norm(
                previous_beta) - np.linalg.norm(beta))
            previous_beta = beta
            i += 1

        return beta, weights

    def predict(self, exposure: np.array) -> np.array:
        return exposure @ self.beta

    def params(self):
        raise Exception("Unimplemented.")

    def vcov(self):
        raise Exception("Unimplemented.")

    def result(self):
        raise Exception("Unimplemented.")
