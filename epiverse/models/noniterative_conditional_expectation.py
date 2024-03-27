from typing import List, Callable
import numpy as np
import pandas as pd
import statsmodels.api as sm

from icecream import ic
from epiverse.models import FunctionModel, LogisticRegression, MultioutcomeRegression
from epiverse.utilities import df_formula_parser, formula_parser_lhs


def noniterative_ice(treatment_strategy: Callable,
                     pooled_confounder_model: FunctionModel,
                     pooled_outcome_model: FunctionModel,
                     eqns: List[List[str]],
                     data: pd.DataFrame,
                     tol: float = 1e-8,
                     batch_size: int = 1000,
                     seed: int = 700,
                     verbose: bool = True) -> np.ndarray:
    ic(eqns)
    err = np.full((len(eqns)), np.inf)
    result = np.zeros((len(eqns)))
    running_events = np.zeros((1, len(eqns)))
    running_batches = 0

    rng = np.random.default_rng(seed=seed)

    while not np.isclose(result, err, atol=tol).all():
        sample = data.sample(n=batch_size, replace=True, axis=0)
        hazard = np.zeros((batch_size, len(eqns)))

        # For each time point, do the simulation
        for i, eqn in enumerate(eqns):
            if i != 0:
                dm_l, dm_l_mat = df_formula_parser(eqn[0], data=sample)
                # Make random draw from predicted values.
                next_confounders = pooled_confounder_model.predict_random(i-1,
                                                                          dm_l_mat, rng=rng).reshape(sample[dm_l.columns].shape)
                sample[[f"pred_{c}" for c in dm_l.columns]] = next_confounders
            # Simulate next treatment
            dm_trt, dm_trt_mat = df_formula_parser(eqn[1], data=sample)
            next_treatment = treatment_strategy(i, dm_trt_mat)
            sample[f"pred_{dm_trt.columns[0]}"] = next_treatment

            # Retrieve hazard of event
            _, dm_haz_mat = df_formula_parser(eqn[2], data=sample)
            predicted_outcome = pooled_outcome_model.predict_random(t=i,
                                                                    values=dm_haz_mat, rng=rng)

            hazard[:, i] = predicted_outcome

        # Accumulate hazards towards getting probability
        never_events = np.all(hazard == 0, axis=1)
        first_events = np.argmax(hazard, axis=1)

        _, first_event_count = np.unique(first_events, return_counts=True)
        # Individuals who never had an event have their max index at 0; subtract them off.
        first_event_count[0] -= never_events.sum()

        running_events += first_event_count
        running_batches += batch_size
        running_denominator = running_batches - \
            np.cumsum(running_events) + running_events[0]
        running_hazard = running_events.flatten() / running_denominator.flatten()
        err = result.copy()
        # TODO: predictions incorrect for normal covariate.
        result = 1 - np.cumprod(1 - running_hazard)

        if (running_batches % (batch_size * 100) == 0) & verbose:
            print(
                f"Batch {running_batches}: current {np.array2string(result, precision=4, floatmode='fixed')}, diff {np.array2string(err - result, precision=4, floatmode='fixed')}")
    if verbose:
        print(
            f"Batch {running_batches}: current {np.array2string(result, precision=4, floatmode='fixed')}, diff {np.array2string(err - result, precision=4, floatmode='fixed')}")

    return result


def noniterative_ice_constructor(treatment_strategy: Callable,
                                 confounder_list: List[List[str]],
                                 treatment_list: List[str],
                                 outcome_list: List[str],
                                 censoring_list: List[str],
                                 data: pd.DataFrame,
                                 pooled_confounder_model: Callable,
                                 pooled_outcome_model: Callable,
                                 id_col: str,
                                 treatment_prefix: str,
                                 confounder_prefixes: List[str],
                                 censoring_prefix: str,
                                 outcome_prefix: str,
                                 tol: float = 1e-9,
                                 batch_size: int = 1000,
                                 verbose=True):
    """A constructor for a common call to noniterative ICE estimator. 

    Args:
        treatment_strategy (Callable): Function which takes in time t and a design matrix, and returns the treatment assignment. 
        confounder_list (List[List[str]]): at each outer index, the list of covariates relevant to that time point, starting at time 0. 
        treatment_list (List[str]): the list of time-dependent treatments for that time point, starting at time 0. 
        outcome_list (List[str]): the list of patsy equations for the outcome strategy specification, starting at time 1. 
        data (pd.DataFrame): the wide dataset that the patsy equations will draw from. 
        id_col (str): the column identifier for grouping by IDs. 
        treatment_prefix (str): the prefix for treatment variables.
        confounder_prefixes (List[str]): the list of confounders to account for. 
        censoring_prefix (str): the prefix for censoring variables.
        outcome_prefix (str): the prefix for outcome variables.
        tol (float, optional): the convergence tolerance. Defaults to 1e-9.
        batch_size (int, optional): how wide to make the batches for MC simulation. Defaults to 1000.

    Returns:
        np.ndarray: resulting array of risk estimates from NICE estimator. 
    """

    # First, handle equation construction
    if len(treatment_list) != len(confounder_list) & len(confounder_list) != len(outcome_list):
        raise Exception(
            f"Lists are not of same lengths; trt: {len(treatment_list)}, con: {len(confounder_list)}, out: {len(outcome_list)}")

    constructed_eqns = []

    running_confounders = ""
    running_treatments = ""
    for i, (cnf, trt, out) in enumerate(zip(confounder_list, treatment_list, outcome_list)):
        eqn_list = []

        cnfs = "+".join(cnf)

        if i == 0:  # First equations can use observed values
            eqn_list.append([])
            eqn_list.append(f"{trt} ~ {cnfs}")
            eqn_list.append(f"{out} ~ pred_{trt} + {cnfs}")
            running_confounders = cnfs
        else:
            eqn_list.append(
                f"{cnfs} ~ {running_treatments} + {running_confounders}")
            eqn_list.append(f"{trt} ~ {'+'.join([f'pred_{c}' for c in cnf])}")
            eqn_list.append(
                f"{out} ~ pred_{trt} + {'+'.join([f'pred_{c}' for c in cnf])}")
            running_confounders = '+'.join([f"pred_{c}" for c in cnf])

        constructed_eqns.append(eqn_list)
        running_treatments = f"pred_{trt}"

    # Then, handle model creation.
    # Data format: need as long format with time, id, current treatment, current covariates, next outcome
    # To do this, first create variables for prior treatments and prior covariates
    stubnames = [treatment_prefix, *confounder_prefixes,
                 censoring_prefix, outcome_prefix]

    longdata = pd.wide_to_long(
        data, stubnames=stubnames, i=id_col, j="t").reset_index()

    shifted = longdata.groupby(id_col).shift(-1)

    longdata[["shift_Y", "shift_C", *[f"shift_{cp}" for cp in confounder_prefixes]]] = shifted[[
        outcome_prefix, censoring_prefix, *confounder_prefixes]]

    # Only take people who are uncensored and nonevents at current time point
    filtered_data = longdata[(longdata["Y"] == 0) & (
        longdata["C"] == 0) & (longdata["shift_C"] == 0)]

    # Create the pooled models to pass forward
    covs_list = [f"{treatment_prefix}", *confounder_prefixes]

    pooled_cf_model = pooled_confounder_model(eqn=f"{'+'.join([f'shift_{cp}' for cp in confounder_prefixes])} ~ {'+'.join(covs_list)}",
                                              data=filtered_data)

    pooled_out_model = pooled_outcome_model(eqn=f"shift_Y ~ {'+'.join(covs_list)}",
                                            data=filtered_data)

    return noniterative_ice(treatment_strategy=treatment_strategy,
                            pooled_confounder_model=pooled_cf_model,
                            pooled_outcome_model=pooled_out_model,
                            eqns=constructed_eqns,
                            data=data,
                            tol=tol,
                            batch_size=batch_size,
                            verbose=verbose)
