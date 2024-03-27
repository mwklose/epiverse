from epiverse.utilities.data_generation import DataGeneratorPregnancy
from epiverse.survival import KaplanMeier, AalenJohansen
from icecream import ic

import numpy as np
import pandas as pd


def make_data_files(start=1, stop=13):
    filelist = [
        f"../Research/Multiple_outcomes_pregnancy/Scenario {i}.xlsx" for i in range(start, stop)]

    for i, fl in enumerate(filelist):
        ic(f"{i}: {fl} starting")
        dgp = DataGeneratorPregnancy(700, parameters_file=fl)
        data: pd.DataFrame = dgp.generate_data(n=1_500_000, verbose=False)
        ic(f"{i}: {fl} generated, saving now")
        data.to_csv(f"runners/preg_sim/pregsimdata_{i+start}.csv")

    return


# make_data_files(start=1, stop=4)


def data_check(start=0, stop=12):
    filelist = [
        f"runners/preg_sim/pregsimdata_{i+1}.csv" for i in range(start, stop)]

    for i, fl in enumerate(filelist):
        data = pd.read_csv(fl)
        trt = data[data["treatment_assignment"] == 1]

        df = trt.groupby("observed_event")["ga"].value_counts()
        df.to_csv(f"tocheck_{i+1}.csv")
        km = KaplanMeier(
            time=trt["ga"], delta=trt["observed_event"], weights=1, event_indicator=1)

        ic(km.survival_estimates)


# data_check(0, 4)


def km_efficacy(start=0, stop=12):
    filelist = [
        f"runners/preg_sim/pregsimdata_{i+1}.csv" for i in range(start, stop)]

    for i, fl in enumerate(filelist):
        data = pd.read_csv(fl)
        # Composite outcome of preeclampsia or fetal death
        for stub in ["observed", "untreated", "treated"]:
            data[f"{stub}_composite"] = ((data[f"{stub}_event"] == 1) | (
                data[f"{stub}_preeclampsia"] == 1)).astype(int)
        data["observed_nc_composite"] = ((data["observed_event_nc"] == 1) | (
            data["observed_preeclampsia_o_nc"] == 1)).astype(int)
        treated = data[data["treatment_assignment"] == 1]
        untreated = data[data["treatment_assignment"] == 0]

        results = {}
        results["observed_trt"] = KaplanMeier(
            time=treated["ga"],
            delta=treated[f"observed_composite"],
            weights=1,
            event_indicator=1
        )

        results["observed_unt"] = KaplanMeier(
            time=untreated["ga"],
            delta=untreated[f"observed_composite"],
            weights=1,
            event_indicator=1
        )

        time = results["observed_trt"].survival_estimates[:, 0]

        r_trt_observed = (1 - results["observed_trt"].survival_estimates[:, 1])
        r_unt_observed = (1 - results["observed_unt"].survival_estimates[:, 1])
        rd_observed = (1 - results["observed_trt"].survival_estimates[:, 1]) - \
            (1 - results["observed_unt"].survival_estimates[:, 1])

        # rr_observed = (1 - results["observed_trt"].survival_estimates[:, 1]) / \
        #     (1 - results["observed_unt"].survival_estimates[:, 1])

        rd_var_observed = results["observed_trt"].survival_estimates[:, 2] + \
            results["observed_unt"].survival_estimates[:, 2]

        observed = pd.DataFrame({"Label": "Observed",
                                 "Time": time,
                                 "rt": r_trt_observed,
                                 "ru": r_unt_observed,
                                #  "rr": rr_observed,
                                 "rd": rd_observed,
                                 "var": rd_var_observed})

        results["counterfactual_trt"] = KaplanMeier(
            time=data["ga_t"],
            delta=data["treated_composite"],
            weights=1,
            event_indicator=1
        )

        results[f"counterfactual_unt"] = KaplanMeier(
            time=data["ga_u"],
            delta=data["untreated_composite"],
            weights=1,
            event_indicator=1
        )

        r_trt_counterfactual = (
            1 - results["counterfactual_trt"].survival_estimates[:, 1])
        r_unt_counterfactual = (
            1 - results["counterfactual_unt"].survival_estimates[:, 1])
        rd_counterfactual = (1 - results["counterfactual_trt"].survival_estimates[:, 1]) - \
            (1 - results["counterfactual_unt"].survival_estimates[:, 1])

        # rr_counterfactual = (1 - results["counterfactual_trt"].survival_estimates[:, 1]) / \
        #     (1 - results["counterfactual_unt"].survival_estimates[:, 1])

        rd_var_counterfactual = results["counterfactual_trt"].survival_estimates[:, 2] + \
            results["counterfactual_unt"].survival_estimates[:, 2]

        counterfactual = pd.DataFrame({"Label": "Counterfactual",
                                       "Time": time,
                                       "rt": r_trt_counterfactual,
                                       "ru": r_unt_counterfactual,
                                       #    "rr": rr_counterfactual,
                                       "rd": rd_counterfactual,
                                       "var": rd_var_counterfactual})

        results["truth_trt"] = KaplanMeier(
            time=treated["ga_o_nc"],
            delta=treated["observed_composite"],
            weights=1,
            event_indicator=1
        )

        results["truth_unt"] = KaplanMeier(
            time=untreated["ga_o_nc"],
            delta=untreated["observed_composite"],
            weights=1,
            event_indicator=1
        )

        r_trt_truth = (
            1 - results["truth_trt"].survival_estimates[:, 1])
        r_unt_truth = (
            1 - results["truth_unt"].survival_estimates[:, 1])
        rd_truth = r_trt_truth - r_unt_truth

        rd_var_truth = results["truth_trt"].survival_estimates[:, 2] + \
            results["truth_unt"].survival_estimates[:, 2]

        truth = pd.DataFrame({"Label": "Truth",
                                       "Time": time,
                                       "rt": r_trt_truth,
                                       "ru": r_unt_truth,
                                       "rd": rd_truth,
                                       "var": rd_var_truth})

        measures = pd.concat([observed, counterfactual, truth], axis=0)

        measures.to_csv(f"runners/preg_sim/results/km_efficacy_{i+1}.csv")


# km_efficacy(start=0, stop=3)


def km_safety():
    pass


def km_preterm():
    pass


def aj_efficacy(start=0, stop=12):
    filelist = [
        f"runners/preg_sim/pregsimdata_{i+1}.csv" for i in range(start, stop)]

    for i, fl in enumerate(filelist):
        data = pd.read_csv(fl)
        for stub in ["observed", "untreated", "treated"]:
            data[f"{stub}_composite"] = np.select(
                condlist=[
                    ((data[f"{stub}_event"] == 1) |
                     (data[f"{stub}_preeclampsia"] == 1)),
                    (data[f"{stub}_event"] == 2),
                    (data[f"{stub}_event"] == 0),
                    True
                ],
                choicelist=[
                    1,
                    2,
                    0,
                    -1
                ])

        treated = data[data["treatment_assignment"] == 1]
        untreated = data[data["treatment_assignment"] == 0]

        times = np.sort(np.unique(data["ga"]))

        aj_observed_trt = AalenJohansen(
            time=treated["ga"],
            delta=treated["observed_composite"],
            weights=1,
            event_indicator=[1, 2]
        )

        aj_observed_unt = AalenJohansen(
            time=untreated["ga"],
            delta=untreated["observed_composite"],
            weights=1,
            event_indicator=[1, 2]
        )

        i_trt = (aj_observed_trt.predict(times, cause=1))[:, 0]
        i_unt = (aj_observed_unt.predict(times, cause=1))[:, 0]

        observed = pd.DataFrame({
            "Label": "Observed",
            "t": times,
            "I_trt": i_trt,
            "I_unt": i_unt,
            "RD": i_trt - i_unt
        })

        aj_counter_trt = AalenJohansen(
            time=data["ga_t"],
            delta=data["treated_composite"],
            weights=1,
            event_indicator=[1, 2]
        )

        aj_counter_unt = AalenJohansen(
            time=data["ga_u"],
            delta=data["untreated_composite"],
            weights=1,
            event_indicator=[1, 2]
        )

        i_trt = (aj_counter_trt.predict(times, cause=1))[:, 0]
        i_unt = (aj_counter_unt.predict(times, cause=1))[:, 0]

        counterfactual = pd.DataFrame({
            "Label": "Counterfactual",
            "t": times,
            "I_trt": i_trt,
            "I_unt": i_unt,
            "RD": i_trt - i_unt
        })

        results = pd.concat([observed, counterfactual], axis=0)
        results.to_csv(f"runners/preg_sim/results/aj_efficacy_{i+1}.csv")


aj_efficacy(start=0, stop=3)


def aj_safety():
    pass


def aj_safety_v2():
    pass


def aj_preterm():
    pass


def marginal_efficacy():
    pass


def marginal_safety():
    pass


def marginal_preterm():
    pass
# Analyses:
# KM Efficacy   - 0=censored or fetal death, 1=severe preeclampsia AND/OR fetal death
# KM Safety,    - 0=censored or fetal death, 1=SGA
# KM Preterm,   - 0=censored or term birth or fetal death, 1=preterm birth
# AJ Efficacy   - 0=censored, 1=severe preeclampsia AND/OR fetal death, 2=live birth
# AJ Safety     - 0=censored, 1=live birth+SGA, 2=live birth+no SGA, 3=fetal death
# AJ Safety v2  - 0=censored OR livebirth, 1=SGA, 2=fetal death
# AJ Preterm    - 0=censored or term birth, 1=preterm birth, 2=fetal death,
# Marginal Efficacy
# Marginal Safety
# Marginal Preterm
