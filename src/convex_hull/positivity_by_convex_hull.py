import pandas as pd
from typing import List


class PositivityConvexHull:

    def __init__(self, treatment_data: pd.DataFrame, nontreatment_data: pd.DataFrame, list_of_variables: List):
        # Items to hold:
        # Original data sources, original list of variables
        # Intersection between the two convex hulls, as polygon
        # Intersection between the two convex hulls, as list of data
        # List of all data not in intersection between both
        # Ability to check additional, future data (ability to update or not)
        pass
