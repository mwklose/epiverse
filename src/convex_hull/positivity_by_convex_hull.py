import pandas as pd
from typing import List


class PositivityConvexHull:

    def __init__(self, treatment_data: pd.DataFrame, nontreatment_data: pd.DataFrame, list_of_variables: List):
        self.treatment_data = treatment_data
        self.nontreatment_data = nontreatment_data
        self.list_of_variables = list_of_variables

        self.treated_convex_hull = self.get_convex_hull(self.treatment_data)
        self.nontreated_convex_hull = self.get_convex_hull(
            self.nontreatment_data)

        # Items to hold:
        # Original data sources, original list of variables
        # Intersection between the two convex hulls, as polygon
        # Intersection between the two convex hulls, as list of data
        # List of all data not in intersection between both
        # Ability to check additional, future data (ability to update or not)

    def get_convex_hull(self, data: pd.DataFrame) -> None:
        pass

    def check_point_for_positivity(self):
        pass

    def generate_list_of_valid_points(self):
        pass
