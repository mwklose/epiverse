import numpy as np
import pandas as pd

import sklearn
from sklearn.linear_model import LogisticRegression
from epiverse.utilities.data_generation.data_generator_binary_confounding_triangle import DataGeneratorBinaryConfoundingTriangle


def paf_simulation(N=1200, N_MULT=10, y1=0.4, y0=0.2, treatment_study=0.5, confounder_study=0.75, treatment_target=0.25, confounder_target=0.25):
    # Odds ratios not equal across study and target, but find solution to when they are equal

    study_pop = DataGeneratorBinaryConfoundingTriangle(
        n=N, treatment_prevalence=treatment_study, confounder_prevalence=confounder_study, y1=y1, y0=y0)
    target_pop = DataGeneratorBinaryConfoundingTriangle(
        n=N * N_MULT, treatment_prevalence=treatment_target, confounder_prevalence=confounder_target, y1=y1, y0=y0)

    # Calculate the PAF the correct way
    # P[Y^0]
    # IPTW
    iptw = LogisticRegression().fit(X=study_pop["W"], y=study_pop["A"])
    print(iptw)
    # IOSW_W

    # Calculate the PAF the incorrect way


if __name__ == "__main__":
    paf_simulation()
