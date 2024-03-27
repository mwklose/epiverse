from typing import Callable
from epiverse.models import FunctionModel
from epiverse.utilities.math.finite_differences import mv_derivative
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from icecream import ic

@dataclass
class Turnbull(FunctionModel): 
    data: pd.DataFrame
    censoring_low: str = "censoring_low"
    censoring_high: str = "censoring_high"
    truncation_low: str = "truncation_low"
    truncation_high: str = "truncation_high"
    epsilon: float = 1e-8
    s: NDArray = field(init=False)
    vcov: NDArray = field(init=False)
    f_estimates: NDArray = field(init=False)


    def __post_init__(self) -> None: 
        self.s = self._fit_procedure()
        
        self.vcov = self._vcov()
        self.f_estimates = np.append(arr=np.zeros(shape=1), values=np.cumsum(a=self.s))
        self.survival_estimates = 1 - self.f_estimates

    def _fit_procedure(self) -> np.ndarray: 
        # TODO: Turnbull and KM do not match because of funkiness with include np.inf as event time. 
        # However, if not include, then row sums of people censored at the end will be 0. 
        unique_exit_times: np.ndarray = np.append(np.unique(self.data.loc[self.data[self.censoring_low] == self.data[self.censoring_high], self.censoring_low]), np.inf)
        unique_enter_times: np.ndarray = np.roll(a=unique_exit_times, shift=1)
        unique_enter_times[0] = 0

        def mu_j(s: np.ndarray): 
            # 1. Find if point between enter and exit. 
            def between(x):  
                # Anything except upper boundary below lower level and lower boundary above upper level
                not_between = (x[self.censoring_high] <= unique_enter_times) | (x[self.censoring_low] > unique_exit_times)
                return np.logical_not(not_between)
            
            df = self.data.apply(between, axis=1)
            stacked_data = np.stack(arrays=df.to_list(), axis=0)
            # 2. Multiply (1) by s to get total density
            density_df = stacked_data.T * s
            sum_df: np.ndarray = stacked_data @ s
            # 3. Divide (2) by sum of (2) 
            prob = np.divide(density_df.T, sum_df, out=np.zeros_like(density_df.T), where=sum_df > 0)
            # 4. Result of IxJ array. 
            return prob

        def nu_j(s: np.ndarray)-> np.ndarray: 
            def not_between(x): 
                not_between_var = (x[self.truncation_high] <= unique_enter_times) | (x[self.truncation_low] > unique_exit_times)
                return not_between_var

            # Note: not_between here, so beta values are flipped relative to Turnbull 1976.
            df: pd.Series = self.data.apply(f=not_between, axis=1)
            stacked_beta: np.ndarray = np.stack(arrays=df.to_list(), axis=0)
            numerator = stacked_beta.T * s
            sum_beta = (1-stacked_beta) @ s
            exp_j = numerator.T / sum_beta
            
            return exp_j


        vec_s = np.ones(shape=(unique_enter_times.shape[0], 1)) / unique_enter_times.shape[0]
        delta_s: np.ndarray = np.full(shape=vec_s.shape, fill_value=np.inf)
        
        # Expectation-Maximization Algorithm
        i = 0 
        while np.all(a=np.abs(delta_s) > self.epsilon): 
            # Expectation: evaluate mu, nu, pi
            mu: np.ndarray = mu_j(s=vec_s)
            nu: np.ndarray = nu_j(s=vec_s)
            
            sj = (mu + nu)
            m = np.sum(a=sj, axis=None)
            
            # Maximization: find new value of s for next iteration
            pi: np.ndarray = np.reshape(np.sum(a=sj, axis=0) / m, (-1, 1))
            ic(mu, nu, pi)
            delta_s = vec_s - pi
            vec_s = pi
            # Stop on convergence to a value
            i += 1
        # Return converged value
        return vec_s

    def _vcov(self) -> NDArray[np.float64]: 
        unique_exit_times: np.ndarray = np.append(np.unique(self.data.loc[self.data[self.censoring_low] == self.data[self.censoring_high], self.censoring_low]), np.inf)
        unique_enter_times: np.ndarray = np.roll(a=unique_exit_times, shift=1)
        unique_enter_times[0] = 0

        # Variance estimation by inverse of second derivative of log-likelihood
        def log_likelihood(s) -> float: 
            def within_censoring(x):  
                # Anything except upper boundary below lower level and lower boundary above upper level
                not_between = (x[self.censoring_high] <= unique_enter_times) | (x[self.censoring_low] > unique_exit_times)
                return np.logical_not(not_between)

            def within_truncation(x):  
                # Anything except upper boundary below lower level and lower boundary above upper level
                not_between = (x[self.truncation_high] <= unique_enter_times) | (x[self.truncation_low] > unique_exit_times)
                return np.logical_not(not_between)
            
            alpha = self.data.apply(within_censoring, axis=1)
            stacked_alpha: np.ndarray = np.stack(arrays=alpha.to_list(), axis=0)
            beta = self.data.apply(within_truncation, axis=1)
            stacked_beta: np.ndarray = np.stack(arrays=beta.to_list(), axis=0)

            log_alpha = np.log(stacked_alpha @ s)
            log_beta = np.log(stacked_beta @ s)
            return np.sum(log_alpha - log_beta)

        second_derivative_matrix: NDArray = mv_derivative(log_likelihood, x0=self.s)
        variance: NDArray[np.float64] = -1 * np.linalg.inv(second_derivative_matrix)
        return variance
    
    def estimate(self) -> Callable: 

        return lambda x: 1


    def predict(self, t: pd.Series) -> np.ndarray: 

        return np.ones(shape=1)


    def predict_random(self, **kwargs) -> np.ndarray:
        return np.random.default_rng(seed=700).binomial(n=1, p=0.25, size=self.data.shape[0])