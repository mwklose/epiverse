import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple

from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.distance import cdist, pdist
import scipy.optimize as sciopt
import scipy.spatial.qhull as qhull


BUFFER = 0.2

# TODO: measure of distance for points outside the convex hulls
# TODO: proportion of points for each hull inside/outside the convex hull.


class PositivityConvexHull:

    def __init__(self, treatment_data: pd.DataFrame, nontreatment_data: pd.DataFrame, list_of_variables: List):
        self.treatment_data = treatment_data
        self.nontreatment_data = nontreatment_data
        self.list_of_variables = list_of_variables

        self.treated_convex_hull = self.get_convex_hull(self.treatment_data)
        self.nontreated_convex_hull = self.get_convex_hull(
            self.nontreatment_data)

        # Assume intersection exists at first.
        self.intersection_flag = True
        self.intersection = None
        self.valid_points = None
        self.invalid_points = None
        self.intersection = self.get_hull_intersection()

    def get_hull_points(self) -> Tuple[pd.DataFrame]:
        treated_hull = self.treatment_data.loc[self.treated_convex_hull.vertices]
        nontreated_hull = self.nontreatment_data.loc[self.nontreated_convex_hull.vertices]

        return treated_hull, nontreated_hull

    def get_convex_hull(self, data: pd.DataFrame) -> ConvexHull:
        covariates = data[self.list_of_variables]
        return ConvexHull(covariates)

    def get_hull_intersection(self) -> ConvexHull:
        if not self.intersection_flag:
            return None
        # If already done intersection, just return that.
        if self.intersection and self.intersection_flag:
            return self.intersection
        # Create list of halfspaces
        treated_halfspaces = self.treated_convex_hull.equations
        nontreated_halfspaces = self.nontreated_convex_hull.equations

        total_halfspace = np.vstack(
            (treated_halfspaces, nontreated_halfspaces))

        self.valid_points, self.invalid_points = self.generate_list_of_valid_points(
            halfspace_equations=total_halfspace)

        if self.valid_points is None and self.invalid_points is None:
            self.intersection_flag = False
            return None

        interior_point = self.valid_points.iloc[0][self.list_of_variables]
        # Perform the intersection
        halfspace_intersection = HalfspaceIntersection(
            total_halfspace, interior_point=interior_point)

        self.intersection = ConvexHull(
            halfspace_intersection.intersections, qhull_options="FN")

        # Return the intersection
        return self.intersection

    def check_point_for_positivity(self, test_point: np.array):
        intersection = self.get_hull_intersection()

        if intersection is None:
            return False

        test_point_intercept = np.insert(test_point, test_point.shape[-1], 1)
        halfspace_checks = intersection.equations @ test_point_intercept

        within_all_halfspaces = np.all(halfspace_checks < 0)

        return within_all_halfspaces

    def get_distance_point_to_hull(self, test_point: np.array, hull: ConvexHull) -> Tuple[float, Tuple]:
        if hull is None:
            raise Exception("Provided hull must be present")
        # Get points on simplices to check
        vertex_points = hull.points[hull.vertices]

        # Use minimization algorithm for convex cmombinations of the vertices.
        constraints = [
            # Convex Constraint - have values add to 1
            sciopt.LinearConstraint(
                np.ones(vertex_points.shape[0]), 1, 1),
            # Constraint: have all individual values be positive, between 0 and 1
            sciopt.LinearConstraint(np.identity(vertex_points.shape[0]), 0, 1)
        ]

        # For now, only compute Euclidean distance.
        def objective_function(x):
            vector_distance = test_point - vertex_points.T @ x

            distance = np.linalg.norm(vector_distance)
            return distance

        initial_guess = [1/vertex_points.shape[0]] * vertex_points.shape[0]

        calculation = sciopt.minimize(
            objective_function, x0=initial_guess, constraints=constraints)

        return calculation.fun, calculation.x

    def generate_distances_of_invalid_points(self, metric: str = "Euclidean") -> pd.DataFrame:
        if not self.intersection:
            self.get_hull_intersection()

        invalid_treated = self.invalid_points[(
            self.invalid_points["Origin"] == "Treated")]
        invalid_untreated = self.invalid_points[(
            self.invalid_points["Origin"] == "Untreated")]

        if metric == "Euclidean":
            invalid_points = self.invalid_points[self.list_of_variables]

            invalid_distances = invalid_points.apply(
                lambda x: self.get_distance_point_to_hull(
                    np.array(x), self.intersection),
                axis=1
            )

            return invalid_distances

        raise Exception(f"Metric: {metric} not defined yet.")

    def generate_list_of_valid_points(self, halfspace_equations: np.array = None) -> pd.DataFrame:

        if self.valid_points is not None and self.invalid_points is not None:
            return self.valid_points, self.invalid_points

        # Obtain interior points (required by HalfspaceIntersection)
        # Add column for intercept; simplifies later dot product operation.
        treated_data = self.treatment_data[self.list_of_variables]
        treated_data.insert(
            len(self.list_of_variables),
            "Intercept", 1)
        untreated_data = self.nontreatment_data[self.list_of_variables]
        untreated_data.insert(
            len(self.list_of_variables),
            "Intercept", 1
        )

        # For each of the halfspaces, see which points satisfy the conditions of all halfspaces,
        # which then means it is on interior of polygon.
        untreated_in_treated_hull = (
            self.treated_convex_hull.equations @ untreated_data.T).T < 0
        untreated_in_treated_hull = np.all(untreated_in_treated_hull, axis=1)
        treated_in_untreated_hull = (
            self.nontreated_convex_hull.equations @ treated_data.T).T < 0
        treated_in_untreated_hull = np.all(treated_in_untreated_hull, axis=1)

        # Special case: no intersections between points
        if not any(untreated_in_treated_hull) or not any(treated_in_untreated_hull):
            try:
                self.intersection = HalfspaceIntersection(
                    halfspace_equations, np.array([0.5, 0.5]))
            except qhull.QhullError:
                # We were right before; no point exists in both halfspaces.
                self.intersection = None
                return None, None

            # By miracle, if this works, return.
            return self.intersection

        self.valid_points = pd.concat([
            self.treatment_data[treated_in_untreated_hull],
            self.nontreatment_data[untreated_in_treated_hull]
        ], axis=0)

        self.valid_points["Origin"] = np.repeat(
            np.array(["Treated", "Untreated"]),
            repeats=[sum(treated_in_untreated_hull),
                     sum(untreated_in_treated_hull)]
        )

        self.invalid_points = pd.concat([
            self.treatment_data[~treated_in_untreated_hull],
            self.nontreatment_data[~untreated_in_treated_hull]
        ], axis=0)

        self.invalid_points["Origin"] = np.repeat(
            np.array(["Treated", "Untreated"]),
            repeats=[sum(~treated_in_untreated_hull),
                     sum(~untreated_in_treated_hull)]
        )

        invalid_point_distances = self.generate_distances_of_invalid_points()

        self.invalid_points["Distance"] = invalid_point_distances.transform(
            lambda x: x[0])

        return self.valid_points, self.invalid_points

    def plot_2d_convex_hull(self, axes=plt, plot_treated: bool = True, plot_untreated: bool = True, alpha=0.5):
        if len(self.list_of_variables) != 2:
            raise Exception("Plotting only defined for 2D hulls.")
        if not plot_treated and not plot_untreated:
            raise Exception(
                "If you don't want to plot anything, why call Plot 2D Convex Hull?")

        treated_hull, untreated_hull = self.get_hull_points()
        c1, c2 = self.list_of_variables
        if plot_treated:
            axes.fill(treated_hull[c1], treated_hull[c2], alpha=alpha)
            axes.scatter(treated_hull[c1], treated_hull[c2], alpha=alpha)
        if plot_untreated:
            axes.fill(untreated_hull[c1], untreated_hull[c2], alpha=alpha)
            axes.scatter(untreated_hull[c1], untreated_hull[c2], alpha=alpha)

    def plot_2d_halfspaces(self, axes=plt, plot_treated: bool = True, plot_untreated: bool = True) -> None:
        if len(self.list_of_variables) != 2:
            raise Exception("Plotting only defined for 2D hulls.")
        if not plot_treated and not plot_untreated:
            raise Exception(
                "If you don't want to plot anything, why call Plot 2D Halfspaces?")

        treated_hull, untreated_hull = self.get_hull_points()

        if plot_treated:
            eqs = self.treated_convex_hull.equations
            minx, maxx, miny, maxy = self.extract_range(treated_hull)
            x_range = np.linspace(minx, maxx, 200)
            y_range = np.linspace(miny, maxy, 200)

            x_range, y_range = np.meshgrid(x_range, y_range)
            nrows, _ = eqs.shape

            for i in range(nrows):
                # -1 is needed because eq is Ax + b leq 0.
                Z = -1 * (eqs[i, 0] * x_range + eqs[i, 1] * y_range)
                axes.contour(x_range, y_range, Z, [
                    eqs[i, 2]], colors="navy", alpha=0.5)

        if plot_untreated:
            eqs = self.nontreated_convex_hull.equations
            minx, maxx, miny, maxy = self.extract_range(untreated_hull)
            x_range = np.linspace(minx, maxx, 200)
            y_range = np.linspace(miny, maxy, 200)

            x_range, y_range = np.meshgrid(x_range, y_range)
            nrows, _ = eqs.shape

            for i in range(nrows):
                # -1 is needed because eq is Ax + b leq 0.
                Z = -1 * (eqs[i, 0] * x_range + eqs[i, 1] * y_range)
                axes.contour(x_range, y_range, Z, [
                    eqs[i, 2]], colors="dodgerblue", alpha=0.5)

    def plot_2d_intersection(self, axes=plt, plot_halfspaces=False):
        if len(self.list_of_variables) != 2:
            raise Exception("Plotting only defined for 2D hulls.")

        if not self.intersection:
            self.intersection = self.get_hull_intersection()

        # To do: handle
        pts = self.sort_ccw(
            self.intersection.points[self.intersection.vertices])
        axes.fill(pts[:, 0],
                  pts[:, 1],
                  alpha=0.5)
        axes.scatter(pts[:, 0],
                     pts[:, 1],
                     alpha=0.5)

        c1, c2 = self.list_of_variables
        colors = {"Treated": 'navy', "Untreated": 'green'}
        axes.scatter(self.valid_points[c1],
                     self.valid_points[c2],
                     c=self.valid_points["Origin"].map(colors),
                     alpha=0.5, s=10)

        if plot_halfspaces:
            eqs = self.intersection.equations
            hull_points = self.intersection.points[self.intersection.vertices]
            hull_points = pd.DataFrame(
                data=hull_points, columns=self.list_of_variables)
            minx, maxx, miny, maxy = self.extract_range(
                hull_points)
            x_range = np.linspace(minx, maxx, 200)
            y_range = np.linspace(miny, maxy, 200)

            x_range, y_range = np.meshgrid(x_range, y_range)
            nrows, _ = eqs.shape

            for i in range(nrows):
                # -1 is needed because eq is Ax + b leq 0.
                Z = -1 * (eqs[i, 0] * x_range + eqs[i, 1] * y_range)
                axes.contour(x_range, y_range, Z, [
                    eqs[i, 2]], colors="navy", alpha=0.5)

    # Helper function for sorting by CCW, for intersections mainly.
    def sort_ccw(self, points: np.array) -> np.array:
        # To sort CCW, find average value, and then do angles between points.
        average_point = np.mean(points, axis=0)

        pts = np.apply_along_axis(lambda pts_entry: np.arctan2(
            *(pts_entry - average_point)), axis=1, arr=points)

        sorted_indices = np.argsort(pts)
        points = points[sorted_indices]
        return points

    # Helper function for extracting min and max ranges for 2D case.
    # TODO: generalize to more dimensions. Good for checking.
    def extract_range(self, hull_points: np.array) -> Tuple:
        min_xrange = np.floor(
            np.min(hull_points[self.list_of_variables[0]])) - BUFFER
        max_xrange = np.ceil(
            np.max(hull_points[self.list_of_variables[0]])) + BUFFER
        min_yrange = np.floor(
            np.min(hull_points[self.list_of_variables[1]])) - BUFFER
        max_yrange = np.ceil(
            np.max(hull_points[self.list_of_variables[1]])) + BUFFER
        return min_xrange, max_xrange, min_yrange, max_yrange
