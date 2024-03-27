from typing import List, Callable
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from icecream import ic
from epiverse.models import FunctionModel, LogisticRegression, MultioutcomeRegression
from epiverse.utilities import df_formula_parser, formula_parser_lhs


def stratified_ice(treatment_strategy: Callable,
                   hazard_eqns: List[str],
                   hazard_eqn_masks: List[np.ndarray],
                   hazard_pred_eqns: List[str],
                   hazard_pred_masks: List[np.ndarray],
                   iterative_eqns: List[List[str]],
                   iterative_eqn_masks: List[List[np.ndarray]],
                   iterative_pred_eqns: List[List[str]],
                   iterative_pred_masks: List[List[np.ndarray]],
                   data: pd.DataFrame):
    # Step 1 and 2: fit model and predict for model.
    for i, (haz_eqn, haz_mask, haz_pred_eqn, haz_pred_eqn_mask) in enumerate(zip(hazard_eqns, hazard_eqn_masks, hazard_pred_eqns, hazard_pred_masks)):
        model = LogisticRegression(haz_eqn, data[haz_mask])

        haz_lhs, _ = df_formula_parser(haz_eqn, data=data[haz_mask])
        _, rhs = df_formula_parser(haz_pred_eqn, data=data[haz_pred_eqn_mask])

        data.loc[haz_pred_eqn_mask,
                 f"pred_{haz_lhs.columns[0]}{i}"] = model.predict(rhs)

    # Then, do iterated expectation part.
    for i, (iter_eqn, iter_eqn_mask, iter_pred, iter_pred_mask) in enumerate(zip(iterative_eqns, iterative_eqn_masks, iterative_pred_eqns, iterative_pred_masks)):
        if i != 0:
            for j, (eqn, mask, pred, pred_mask) in enumerate(zip(iter_eqn, iter_eqn_mask, iter_pred, iter_pred_mask)):
                iter_model = LogisticRegression(eqn=eqn, data=data[mask])

                iter_lhs, _ = df_formula_parser(eqn, data=data[mask])
                _, iter_pred_rhs = df_formula_parser(
                    pred, data=data[pred_mask])

                predicted_hazard = iter_model.predict(iter_pred_rhs)
                previous_formula = iterative_eqns[i-j-1][0]
                previous_timestep = (df_formula_parser(
                    previous_formula, data=data))[0].columns[0]
                previous_hazard = data.loc[pred_mask, previous_timestep]

                next_col_name = f"{iter_lhs.columns[0][:-1]}{len(iter_eqn) - j - 1}"

                data.loc[pred_mask, next_col_name] = predicted_hazard * \
                    (1-previous_hazard) + previous_hazard

    outvars = [f"pred_{formula_parser_lhs(he)}0" for he in hazard_eqns]

    result = np.mean(data[outvars], axis=0)
    return result


def stratified_ice_constructor(treatment_strategy: Callable,
                               confounder_list: List[List[str]],
                               treatment_list: List[str],
                               outcome_list: List[str],
                               censoring_list: List[str],
                               data: pd.DataFrame,
                               outcome_prefix: str):
    # For Step 1
    y_lkplus = []
    y_lkplus_mask = []
    y_lkplus_pred = []
    y_lkplus_pred_mask = []
    # For Step 3
    y_lk = []
    y_lk_mask = []
    y_lk_pred = []
    y_lk_pred_mask = []

    running_treatment = ""
    running_confounders = ""
    final_outcome = outcome_list[-1]

    for i, (cnf, trt, out, cen) in enumerate(zip(confounder_list, treatment_list, outcome_list, censoring_list)):
        if running_treatment == "":
            # For first iteration through, sometimes slightly different
            y_lkplus.append(f"{out} ~ {'+'.join(cnf)}")
            # People treated concordant with treatment strategy
            lhs, rhs = df_formula_parser(f"{trt} ~ {'+'.join(cnf)}", data=data)

            predicted_treatment = treatment_strategy(i, rhs)
            treatment_concordance = (lhs == predicted_treatment).to_numpy()

            _, rhs = df_formula_parser(f"{cen} - 1", data=data)
            uncensored = (rhs == 0).to_numpy()

            # Add mask for fitting outcome in Step 1
            y_lkplus_mask.append(
                (treatment_concordance & uncensored).copy().flatten())
            # Add column for predicted treatment
            data[f"pred_{trt}"] = predicted_treatment

            # Add eqns for predicted outcome
            y_lkplus_pred.append(f"{'+'.join(cnf)}")

            # Add mask for predicting outcome; all possible in first time step
            y_lkplus_pred_mask.append(np.full(data.shape[0], True))

            # For Step 3: at time point 1, trivial answer.
            y_lk.append([f"pred_{out}0 ~ pred_{out}0 - 1"])
            y_lk_mask.append(np.full(data.shape[0], True))
            y_lk_pred.append([])
            y_lk_pred_mask.append(np.full(data.shape[0], True))

            running_treatment = trt
            running_confounders = '+'.join(cnf)
        else:
            # For first iteration through, sometimes slightly different
            y_lkplus.append(
                f"{out} ~ {'+'.join(cnf)} + {running_confounders}")
            # People treated concordant with treatment strategy
            lhs, rhs = df_formula_parser(
                f"{trt} ~ {running_treatment} + {'+'.join(cnf)} + {running_confounders}", data=data)

            predicted_treatment = treatment_strategy(i, rhs)
            treatment_concordance = (lhs == predicted_treatment).to_numpy()

            uncensored = (data[cen] == 0).to_numpy().flatten()

            # Add mask for fitting outcome in Step 1
            y_lkplus_mask.append(
                (y_lkplus_mask[i-1] & treatment_concordance.flatten() & uncensored).copy().flatten())

            # Add column for predicted treatment
            data[f"pred_{trt}"] = predicted_treatment

            # Add eqns for predicted outcome; needs specific naming scheme
            y_lkplus_pred.append(
                f"{'+'.join(cnf)} + {running_confounders}")

            # Add mask for predicting outcome; all possible in first time step
            y_lkplus_pred_mask.append(y_lkplus_mask[i-1])

            # For Step 3: at later time points, less trivial.
            y_lk_list = []
            y_lk_mask_list = []
            y_lk_pred_list = []
            y_lk_pred_mask_list = []
            for j in range(i, 0, -1):
                y_lk_list.append(
                    f"pred_{out}{j} ~ {' + '.join([' + '.join(k) for k in confounder_list[0:j]][::-1])}")
                y_lk_mask_list.append(y_lkplus_mask[j-1])
                y_lk_pred_list.append(y_lkplus_pred[j-1])
                y_lk_pred_mask_list.append(y_lkplus_pred_mask[j-1])

            y_lk.append(y_lk_list)
            y_lk_mask.append(y_lk_mask_list)
            y_lk_pred.append(y_lk_pred_list)
            y_lk_pred_mask.append(y_lk_pred_mask_list)

            running_treatment = f"{trt} + {running_treatment}"
            running_confounders = f"{'+'.join(cnf)} + {running_confounders}"
    return stratified_ice(treatment_strategy=treatment_strategy,
                          hazard_eqns=y_lkplus,
                          hazard_eqn_masks=y_lkplus_mask,
                          hazard_pred_eqns=y_lkplus_pred,
                          hazard_pred_masks=y_lkplus_pred_mask,
                          iterative_eqns=y_lk,
                          iterative_eqn_masks=y_lk_mask,
                          iterative_pred_eqns=y_lk_pred,
                          iterative_pred_masks=y_lk_pred_mask,
                          data=data)
