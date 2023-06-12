import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from epiverse.utilities.data_generation.data_generator_binary_confounding_triangle import DataGeneratorBinaryConfoundingTriangle


def paf_simulation(N=1200, N_MULT=10, y1=0.4, y0=0.2, treatment_study=0.5, confounder_study=0.75, treatment_target=0.25, confounder_target=0.25):
    # Odds ratios not equal across study and target, but find solution to when they are equal

    study_pop = DataGeneratorBinaryConfoundingTriangle(
        n=N, treatment_prevalence=treatment_study, confounder_prevalence=confounder_study, y1=y1, y0=y0).generate_data()
    target_pop = DataGeneratorBinaryConfoundingTriangle(
        n=N * N_MULT, treatment_prevalence=treatment_target, confounder_prevalence=confounder_target, y1=y1, y0=y0).generate_data()

    # IPTW
    iptw_model = smf.logit("A ~ W", study_pop)
    iptw = iptw_model.fit()

    study_pop["PS"] = iptw.predict()
    # Only among people not treated - each untreated person stands in for all people.
    study_pop["IPTW"] = 1 / (1 - study_pop["A"]) * (1 - study_pop["PS"])

    # IOSW_W
    stacked_dataset = pd.concat(
        [study_pop, target_pop], keys=[1, 0]).reset_index()

    stacked_dataset = stacked_dataset.rename(columns={"level_0": "pop"})

    iosw_w_model = smf.logit("pop ~ W", stacked_dataset)
    iosw_w = iosw_w_model.fit()

    # Exponentiating logit yields odds, and we want inverse odds weighting.
    stacked_dataset["IOSW_W"] = 1 / np.exp(iosw_w.predict(which="linear"))

    # IOSW_AW
    iosw_aw_model = smf.logit("pop ~ W + A", data=stacked_dataset)
    iosw_aw = iosw_aw_model.fit()

    stacked_dataset["IOSW_AW"] = 1 / np.exp(iosw_aw.predict(which="linear"))

    # Calculate PAF the correct way
    y0 = stacked_dataset.loc[stacked_dataset["pop"] == 1] \
        .loc[stacked_dataset["A"] == 0] \

    y0_value = np.average(y0["Y"], weights=y0["IOSW_W"]/y0["IPTW"])

    y = stacked_dataset.loc[stacked_dataset["pop"] == 1]
    y_value = np.average(y["Y"], weights=y["IOSW_AW"])

    paf_true = 1 - y0_value/y_value

    # Calculate the PAF the incorrect way
    y0_incorrect = np.average(y0["Y"], weights=y0["IPTW"])
    y_incorrect = np.average(y["Y"])

    paf_false = 1 - y0_incorrect/y_incorrect
    print(
        f"PAF: {paf_true: .3f} WRONG: {paf_false: .3f} DIFF: {paf_true - paf_false}")


if __name__ == "__main__":
    paf_simulation()
