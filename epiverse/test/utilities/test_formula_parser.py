import pandas as pd
from epiverse.utilities import df_formula_parser


def test_formulas():
    formulas = [
        "x",
        "x + y",
        "y ~ x",
        "y~x",
        "y + y2 ~ x",
        "y+y2 ~ x + z",
        "y ~ x - 1"
    ]

    data = pd.DataFrame({
        "y": [1, 0, 1, 0, 1, 0],
        "y2": [1, 1, 1, 0, 0, 0],
        "x": [0, 1, 1, 0, 0, 1],
        "z": [0.5, 1.5, 1, 0, 2, 0.5],
        "Intercept": 1
    })

    lhs_results = [None, None,
                   data[["y"]], data[["y"]],
                   data[["y", "y2"]], data[["y", "y2"]],
                   data[["y"]]
                   ]
    rhs_results = [data[["Intercept", "x"]], data[["Intercept", "x", "y"]],
                   data[["Intercept", "x"]], data[["Intercept", "x"]],
                   data[["Intercept", "x"]], data[["Intercept", "x", "z"]],
                   data[["x"]]]

    for i, fo in enumerate(formulas):
        lhs, rhs = df_formula_parser(fo, data=data)
        assert all(rhs == rhs_results[i]
                   ), f"Was {rhs}, expected {rhs_results[i]}"
        if lhs is None:
            assert lhs == lhs_results[i], f"Was {lhs}, expected {lhs_results[i]}"
            continue

        assert all(
            lhs == lhs_results[i]), f"Was {lhs}, expected {lhs_results[i]}"
