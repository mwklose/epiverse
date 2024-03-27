import pandas as pd
from pandas import DataFrame, Series
from typing import Callable, Dict, Generator
from numpy.typing import NDArray
import numpy as np
from scipy.special import expit
from icecream import ic


class DataGeneratorPregnancy():

    def __init__(self, seed: int, parameters_file: str = "epiverse/data/Simulation parameters(1).xlsx"):
        self.seed: int = seed
        self.rng: Generator = np.random.default_rng(seed=self.seed) # type: ignore

        self.pd_untreated: DataFrame = pd.read_excel(
            io=parameters_file, sheet_name="Phase1")
        self.pd_treated: DataFrame = pd.read_excel(
            io=parameters_file, sheet_name="Phase2")
        self.pd_preeclampsia: DataFrame = pd.read_excel(
            io=parameters_file, sheet_name="Phase3")
        self.pd_sga: DataFrame = pd.read_excel(
            io=parameters_file, sheet_name="Phase4")
        self.pd_censoring: DataFrame = pd.read_excel(
            io=parameters_file, sheet_name="Phase5")

    def generate_data(self, n: int, total_ga_weeks: int = 41, verbose: bool = False) -> pd.DataFrame:

        df = pd.DataFrame({
            "id": range(n),
            "ga": [[i for i in range(total_ga_weeks)] for id in range(n)],
            "always": 1,
            "never": 0
        })

        df["index_date"] = self._generate_index_date(n=len(df.index))

        df: DataFrame = df.explode(column="ga")

        df[["treated_fetaldeath", "treated_livebirth", "treated_contpreg"]
           ] = self._generate_birth_outcomes_treated(df["ga"])

        df[["untreated_fetaldeath", "untreated_livebirth", "untreated_contpreg"]
           ] = self._generate_birth_outcomes_untreated(df["ga"])

        # Being observed in trial is because of untreated pregnancy continuing until index date
        # Fetal death before index date = not observed.
        possible_early_death: Series[bool] = df["untreated_fetaldeath"] & (df["ga"] < df["index_date"])
        early_death_index = possible_early_death.groupby(
            df.index).cumsum()

        df["fetal_death_before_index"] = early_death_index

        # Trick: because obs_preg is 1 when followed, but 0 following event, cumlative sum will be equal to index date at index date.
        # This mask filters out so only observed 40 week stretches of pregnancy; does not care about event times (yet).
        seen_pregnancy_mask = (df["ga"] >= df["index_date"]) & (
            df["fetal_death_before_index"] == 0)
        seen_pregnancies = df[seen_pregnancy_mask].copy()

        # If interested in individuals who did not reach index date
        unseen_pregnancy_mask = (df["ga"] < df["index_date"]) | (
            df["fetal_death_before_index"] != 0)
        unseen_pregnancies: DataFrame = df[unseen_pregnancy_mask].copy()

        # # TODO: possibly do this at a later point?
        # unseen_pregnancies_summary = unseen_pregnancies.groupby("id").agg({
        #     "ga": "max",
        #     "index_date": "min",
        #     "untreated_fetaldeath": "last",
        #     "untreated_livebirth": "last",
        #     "untreated_contpreg": "last"
        # })

        # Have treatment assignment... with context due to different types of treatment assignments to be handled later
        with pd.option_context('mode.chained_assignment', None):
            seen_pregnancies["treatment_assignment"] = self._generate_treatment_arm(
                seen_pregnancies, id_var="id")

        # Phase 3: preeclampsia symptoms and treatment regeneration
        seen_pregnancies["treated_preeclampsia"] = self._generate_preeclampsia(
            t=seen_pregnancies["ga"],
            treatment=seen_pregnancies["always"])

        seen_pregnancies["untreated_preeclampsia"] = self._generate_preeclampsia(
            t=seen_pregnancies["ga"],
            treatment=seen_pregnancies["never"])

        seen_pregnancies["observed_preeclampsia"] = seen_pregnancies["treated_preeclampsia"] * seen_pregnancies["treatment_assignment"] + \
            seen_pregnancies["untreated_preeclampsia"] * \
            (1 - seen_pregnancies["treatment_assignment"])

        # Regenerate outcomes based on preeclampsia status; treatment is to induce birth.
        seen_pregnancies[["treated_fetaldeath_preeclampsia", "treated_livebirth_preeclampsia"]] = self._regenerate_preeclampsia_outcomes(
            t=seen_pregnancies["ga"],
            observed=seen_pregnancies[[
                "treated_fetaldeath", "treated_livebirth"]],
            preeclampsia=seen_pregnancies["treated_preeclampsia"]
        )

        seen_pregnancies[["untreated_fetaldeath_preeclampsia", "untreated_livebirth_preeclampsia"]] = self._regenerate_preeclampsia_outcomes(
            t=seen_pregnancies["ga"],
            observed=seen_pregnancies[[
                "untreated_fetaldeath", "untreated_livebirth"]],
            preeclampsia=seen_pregnancies["untreated_preeclampsia"]
        )

        seen_pregnancies["treated_sga"] = self._generate_fetal_sga(
            livebirth=seen_pregnancies["treated_livebirth_preeclampsia"],
            treatment=seen_pregnancies["always"],
            preeclampsia=seen_pregnancies["treated_preeclampsia"]
        )

        seen_pregnancies["untreated_sga"] = self._generate_fetal_sga(
            livebirth=seen_pregnancies["untreated_livebirth_preeclampsia"],
            treatment=seen_pregnancies["never"],
            preeclampsia=seen_pregnancies["untreated_preeclampsia"]
        )

        # TODO: don't love the logic here; easy to think that SGA=0 when fetal death.
        treat_sga: Series[int] = (seen_pregnancies["treated_sga"] *
                     seen_pregnancies["treatment_assignment"]).fillna(0)
        untreat_sga: Series[int] = (seen_pregnancies["untreated_sga"] *
                       (1 - seen_pregnancies["treatment_assignment"])).fillna(0)
        seen_pregnancies["observed_sga"] = treat_sga + untreat_sga

        # Always and Never not needed anymore;
        seen_pregnancies: DataFrame = seen_pregnancies.drop(["always", "never"], axis=1)

        # Now, have observed outcomes using Consistency statement
        seen_pregnancies["observed_fetaldeath"] = seen_pregnancies["treated_fetaldeath_preeclampsia"] * seen_pregnancies["treatment_assignment"] + \
            seen_pregnancies["untreated_fetaldeath_preeclampsia"] * \
            (1 - seen_pregnancies["treatment_assignment"])

        seen_pregnancies["observed_livebirth"] = seen_pregnancies["treated_livebirth_preeclampsia"] * seen_pregnancies["treatment_assignment"] + \
            seen_pregnancies["untreated_livebirth_preeclampsia"] * \
            (1 - seen_pregnancies["treatment_assignment"])

        # Note: not handling cases where we randomize women to have preeclampsia or not.
        # Uninformative censoring
        seen_pregnancies["censoring_event"] = self._generate_censoring(
            t=seen_pregnancies["ga"])

        censoring_summary: DataFrame = seen_pregnancies["censoring_event"].groupby(
            by=seen_pregnancies.index).aggregate(func=[np.argmax])

        seen_pregnancies["treated_event_nc"] = self._generate_event_indicator(
            np.zeros(shape=seen_pregnancies["censoring_event"].shape),
            seen_pregnancies["treated_fetaldeath_preeclampsia"],
            seen_pregnancies["treated_livebirth_preeclampsia"]
        )

        seen_pregnancies["treated_event"] = self._generate_event_indicator(
            seen_pregnancies["censoring_event"],
            seen_pregnancies["treated_fetaldeath_preeclampsia"],
            seen_pregnancies["treated_livebirth_preeclampsia"]
        )

        seen_pregnancies["untreated_event_nc"] = self._generate_event_indicator(
            np.zeros(shape=seen_pregnancies["censoring_event"].shape),
            seen_pregnancies["untreated_fetaldeath_preeclampsia"],
            seen_pregnancies["untreated_livebirth_preeclampsia"]
        )

        seen_pregnancies["untreated_event"] = self._generate_event_indicator(
            seen_pregnancies["censoring_event"],
            seen_pregnancies["untreated_fetaldeath_preeclampsia"],
            seen_pregnancies["untreated_livebirth_preeclampsia"]
        )

        seen_pregnancies["observed_event_nc"] = self._generate_event_indicator(
            np.zeros(shape=seen_pregnancies["censoring_event"].shape),
            seen_pregnancies["observed_fetaldeath"],
            seen_pregnancies["observed_livebirth"]
        )

        seen_pregnancies["observed_event"] = self._generate_event_indicator(
            seen_pregnancies["censoring_event"],
            seen_pregnancies["observed_fetaldeath"],
            seen_pregnancies["observed_livebirth"]
        )

        # Lastly, reformat so it is in survival format (somewhat).
        untreated_preg: DataFrame = ((seen_pregnancies.loc[seen_pregnancies.untreated_event >= 0])
                          .sort_values(by=["id", "ga"], kind="stable")
                          .groupby(by="id")
                          .head(n=1))
        untreated_nc_preg: DataFrame = ((seen_pregnancies.loc[seen_pregnancies.untreated_event_nc >= 0])
                             .sort_values(by=["id", "ga"], kind="stable")
                             .groupby(by="id")
                             .head(n=1))

        treated_preg: DataFrame = ((seen_pregnancies.loc[seen_pregnancies.treated_event >= 0])
                        .sort_values(by=["id", "ga"], kind="stable")
                        .groupby(by="id")
                        .head(n=1))

        treated_nc_preg: DataFrame = ((seen_pregnancies.loc[seen_pregnancies.treated_event_nc >= 0])
                           .sort_values(by=["id", "ga"], kind="stable")
                           .groupby(by="id")
                           .head(n=1))

        observed_preg: DataFrame = ((seen_pregnancies.loc[seen_pregnancies.observed_event >= 0])
                         .sort_values(by=["id", "ga"], kind="stable")
                         .groupby(by="id")
                         .head(n=1))

        observed_nc_preg: DataFrame = ((seen_pregnancies.loc[seen_pregnancies.observed_event_nc >= 0])
                            .sort_values(by=["id", "ga"], kind="stable")
                            .groupby(by="id")
                            .head(n=1))

        base_varlist: list[str] = ["id", "index_date", "treatment_assignment", "ga"]
        stubs: list[str] = ["event", "preeclampsia", "sga"]
        stubs_nc: list[str] = ["event_nc", "preeclampsia", "sga"]
        untreated_varlist: list[str] = base_varlist + \
            [f"untreated_{i}" for i in stubs]
        untreated_nc_varlist: list[str] = base_varlist + \
            [f"untreated_{i}" for i in stubs_nc]
        treated_varlist: list[str] = base_varlist + \
            [f"treated_{i}" for i in stubs]
        treated_nc_varlist: list[str] = base_varlist + \
            [f"treated_{i}" for i in stubs_nc]
        observed_varlist: list[str] = base_varlist + \
            [f"observed_{i}" for i in stubs]
        observed_nc_varlist: list[str] = base_varlist + \
            [f"observed_{i}" for i in stubs_nc]

        preg: DataFrame = pd.merge(left=untreated_preg[untreated_varlist],
                        right=treated_preg[treated_varlist],
                        how="left",
                        on=["id", "index_date", "treatment_assignment"], suffixes=["_u", "_t"])
        preg = pd.merge(left=preg,
                        right=observed_preg[observed_varlist],
                        how="left",
                        on=["id", "index_date", "treatment_assignment"], suffixes=["", "_o"])

        preg = pd.merge(left=preg,
                        right=untreated_nc_preg[untreated_nc_varlist],
                        how="left",
                        on=["id", "index_date", "treatment_assignment"], suffixes=["", "_u_nc"])

        preg = pd.merge(left=preg,
                        right=treated_nc_preg[treated_nc_varlist],
                        how="left",
                        on=["id", "index_date", "treatment_assignment"], suffixes=["", "_t_nc"])

        preg = pd.merge(left=preg,
                        right=observed_nc_preg[observed_nc_varlist],
                        how="left",
                        on=["id", "index_date", "treatment_assignment"], suffixes=["", "_o_nc"])

        preg["censor_date"] = censoring_summary

        if verbose:
            seen_pregnancies.to_csv(path_or_buf="seen_pregnancies.csv")
            unseen_pregnancies.to_csv(path_or_buf="unseen_pregnancies.csv")
    
        return preg

    def _generate_birth_outcomes_treated(self, t) -> NDArray:
        pvals: Series[float] = self.pd_treated.iloc[t.to_list(), :].loc[:, ["p_fetaldeath_next",
                                                             "p_livebirth_next", "p_contpreg_next"]]
        outcome: NDArray[int] = self.rng.multinomial( # type: ignore
            1, pvals=pvals, size=len(t.index))

        return outcome

    def _generate_birth_outcomes_untreated(self, t):
        pvals = self.pd_untreated.iloc[t.to_list(), :].loc[:, ["p_fetaldeath_next",
                                                               "p_livebirth_next", "p_contpreg_next"]]
        outcome = self.rng.multinomial(
            1, pvals=pvals, size=len(t.index))

        return outcome

    def _generate_index_date(self, n: int):
        # Fixed 1/17; take the first nonzero observation
        res = np.argmax(self.rng.binomial(
            n=1, p=self.pd_untreated["p_index"], size=(n, self.pd_untreated.shape[0])),
            axis=1)
        return res

    def _generate_treatment_arm(self, data, assignment_type="random_constant", **kwargs) -> pd.Series:
        if assignment_type == "random_constant":
            id_var = kwargs["id_var"]
            unique_ids = data.loc[:, id_var].copy().unique()
            treatment_assignment = self.rng.permutation(
                x=len(unique_ids)) % 2

            treatment_map = {
                k: v for k, v in zip(unique_ids, treatment_assignment)
            }

            treatment = data.loc[:, id_var].copy().map(treatment_map)

            return list(treatment)
        elif assignment_type == "random_stepdown":
            raise Exception("Treatment randomization type not implemented.")
        elif assignment_type == "random_stepup":
            raise Exception("Treatment randomization type not implemented.")
        elif assignment_type == "random_timevarying":
            raise Exception("Treatment randomization type not implemented.")
        # TODO: can do step down, step up, truly random time-varying, etc.
        else:
            raise Exception("Treatment randomization type not available.")

    def _generate_preeclampsia(self, t, treatment, preeclampsia_var="ln_odds_preeclampsia", treatment_effect_var="treat_effect_ln_OR"):
        result = np.zeros(shape=t.shape)
        valid_t = t[t >= 24] - 24
        valid_treatment = np.reshape(treatment[t >= 24], (-1, 1))

        preeclampsia_logodds = self.pd_preeclampsia.iloc[valid_t.to_list(), :].loc[:, [
            preeclampsia_var]]

        treatment_logodds = self.pd_preeclampsia.iloc[valid_t.to_list(), :].loc[:, [
            treatment_effect_var]]

        preeclampsia_logodds_with_treatment = preeclampsia_logodds.values + \
            valid_treatment * treatment_logodds.values

        p = expit(preeclampsia_logodds_with_treatment)
        outcome = self.rng.binomial(1, p=p)
        result[t >= 24] = outcome.flatten()
        return result

    def _regenerate_preeclampsia_outcomes(self, t, observed, preeclampsia, fetaldeath_var="p_fetaldeath"):
        same_result = observed.copy()
        # Because preeclampsia sheet does not go below 24, need to do some fun work.
        valid_t = t[t >= 24] - 24
        valid_preeclampsia = preeclampsia[t >= 24]
        valid_observed = observed[t >= 24]

        fetaldeath_p = self.pd_preeclampsia.iloc[valid_t.to_list(), :].loc[:, [
            fetaldeath_var]].fillna(0)

        fetaldeath_p["livebirth"] = 1 - fetaldeath_p

        preeclampsia_outcome = self.rng.multinomial(
            1, pvals=fetaldeath_p, size=len(valid_t.index))

        no_preeclampsia = (
            1-valid_preeclampsia).to_frame().values * valid_observed
        with_preeclampsia = valid_preeclampsia.to_frame().values * preeclampsia_outcome

        outcome = no_preeclampsia + with_preeclampsia
        same_result[t >= 24] = outcome

        return same_result

    def _generate_fetal_sga(self, livebirth, treatment, preeclampsia, treatment_var="trt_value", preeclampsia_var="preeclampsia_flag", sga_var="p_sga"):
        mydata = pd.DataFrame({
            treatment_var: treatment,
            preeclampsia_var: preeclampsia
        })

        mydata = mydata.merge(
            self.pd_sga, how="left", on=[treatment_var, preeclampsia_var])

        sga = self.rng.binomial(1, p=mydata[sga_var])

        sga_result = np.where(livebirth == 1, sga, np.nan)

        return sga_result

    def _generate_censoring(self, t, censoring_var: str = "p_censoring"):
        censoring_p = self.pd_censoring.iloc[t.tolist(), :].loc[:, [
            censoring_var]].fillna(0)

        censoring_outcome = self.rng.binomial(1, p=censoring_p)
        return censoring_outcome

    def _generate_event_indicator(self, censoring, *args):
        event_indicator = (0 * censoring) - 1
        event_indicator += censoring
        # Then, go through each of args, and add each arg as a new event type.
        for i, arg in enumerate(args):
            event_indicator += (i+2) * arg

        # Double check if censored, set to 0.
        event_indicator *= (1 - censoring)

        return event_indicator
