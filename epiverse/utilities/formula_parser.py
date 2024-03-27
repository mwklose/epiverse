import pandas as pd
from typing import Tuple, List


def formula_parser_lhs(eqn: str) -> List[str]:
    if eqn is None or len(eqn) == 0:
        raise Exception("Must provide equation.")

    components = eqn.split("~")

    if len(components) > 2:
        raise Exception(f"Found {len(components)} sides for equation.")
    if len(components) == 1:
        return None

    if "+" in components[0]:
        lhs_vars = [s.strip() for s in components[0].split("+")]
        lhs_vars = list(filter(None, lhs_vars))
        return lhs_vars
    return components[0].strip()


def formula_parser_rhs(eqn: str) -> List[str]:
    if eqn is None or len(eqn) == 0:
        raise Exception("Must provide equation.")

    components = eqn.split("~")

    if len(components) > 2:
        raise Exception(f"Found {len(components)} sides for equation.")
    if len(components) == 1:
        return None

    if "+" in components[1]:
        rhs_vars = [s.strip()
                    for s in components[1].replace("-", "+").split("+")]
        remove_intercept = any([rv in "1" for rv in rhs_vars])
        if remove_intercept:
            rhs_vars.remove("1")
        rhs_vars = list(filter(None, rhs_vars))
        return rhs_vars
    return components[1]


def df_formula_parser(eqn: str, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if eqn is None or len(eqn) == 0:
        raise Exception("Must provide equation.")

    components = eqn.split("~")

    if len(components) > 2:
        raise Exception(f"Found {len(components)} sides for equation.")
    elif len(components) == 2:
        return df_formula_parser_lhs(components[0], data), df_formula_parser_rhs(components[1], data)
    elif len(components) == 1:
        return None, df_formula_parser_rhs(eqn, data)
    else:
        raise Exception(
            f"Invalid number of components somehow; is {components}")


def df_formula_parser_lhs(eqn: str, data: pd.DataFrame):
    # Assumes receiving only components on left hand side of equation.
    lhs_vars = [s.strip() for s in eqn.split("+")]
    lhs_vars = list(filter(None, lhs_vars))
    lhs_vars_notin_data = filter(
        lambda var: var not in data.columns, lhs_vars)

    if len(list(lhs_vars_notin_data)) != 0:
        raise Exception(
            f"Found variables which do not exist: {list(lhs_vars_notin_data)}")

    return data[lhs_vars]


def df_formula_parser_rhs(eqn: str, data: pd.DataFrame):
    # Assumes right hand side of equation is components and possibly intercept.
    remove_intercept = any([_ in eqn for _ in ["-1", "- 1"]])
    rhs_vars = [s.strip() for s in eqn.replace("-", "+").split("+")]

    if remove_intercept:
        rhs_vars.remove("1")

    rhs_vars = list(filter(None, rhs_vars))
    rhs_vars_notin_data = filter(
        lambda var: var not in data.columns, rhs_vars)
    if len(list(rhs_vars_notin_data)) != 0:
        raise Exception(
            f"Found variables which do not exist: {list(rhs_vars_notin_data)}")

    if remove_intercept:
        return data[rhs_vars]
    data["Intercept"] = 1
    return data[["Intercept", *rhs_vars]]
