from epiverse.utilities.dag import DAG
from epiverse.utilities.data_generation import DataGeneratorDAG
from scipy.special import logit, expit
import numpy as np
import pandas as pd
import statsmodels.api as sm
from epiverse.models import AIPW


def generate_data():
    N = 1000
    A_PROP = 0.5

    rng = np.random.default_rng()

    def l_func():
        return rng.normal(0, 2, N)

    def a_func(l):
        logit_a = logit(A_PROP)
        a_prop = logit_a + 0.25 * (l - np.mean(l))
        return rng.binomial(1, expit(a_prop), N)

    def y_func(a, l):
        mu_y = 2 * a + 0.15 * l**2
        return rng.normal(mu_y, 2, N)

    dag = DAG(node_dict={"L": l_func,
                         "A": a_func,
                         "Y": y_func},
              edge_list=[["A", "Y"], ["L", "Y"], ["L", "A"]])

    data = DataGeneratorDAG(dag=dag).generate_data()
    return data


def main():
    data = generate_data()
    data["intercept"] = 1
    data["a0"] = 0
    data["a1"] = 1
    data["l**2"] = data["L"]**2

    misspecified_exposure_model = sm.GLM(
        endog=data["A"], exog=data["intercept"], family=sm.families.Binomial())
    misspecified_exposure = misspecified_exposure_model.fit()

    data["misspecified_a"] = misspecified_exposure.predict()

    correct_exposure_model = sm.GLM(
        endog=data["A"], exog=data[["intercept", "L"]], family=sm.families.Binomial())
    correct_exposure = correct_exposure_model.fit()
    data["correct_a"] = correct_exposure.predict()

    misspecified_outcome_model = sm.GLM(
        endog=data["Y"], exog=data[["A", "L"]])
    misspecified_outcome = misspecified_outcome_model.fit()
    data["misspecified_y"] = misspecified_outcome.predict()

    correct_outcome_model = sm.GLM(
        endog=data["Y"], exog=data[["A", "L", "l**2"]])
    correct_outcome = correct_outcome_model.fit()
    data["correct_y"] = correct_outcome.predict()

    data["misspecified_y0"] = misspecified_outcome.predict(
        exog=data[["a0", "L"]])
    data["misspecified_y1"] = misspecified_outcome.predict(
        exog=data[["a1", "L"]])

    data["correct_y0"] = correct_outcome.predict(
        exog=data[["a0", "L", "l**2"]])
    data["correct_y1"] = correct_outcome.predict(
        exog=data[["a1", "L", "l**2"]])

    dr_a1 = data["A"] * data["Y"] / data[""]


if __name__ == "__main__":
    main()
