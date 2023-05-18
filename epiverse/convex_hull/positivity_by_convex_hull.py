import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from scipy.spatial import ConvexHull, HalfspaceIntersection
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

        self.valid_points = None
        self.invalid_points = None
        self.intersection = None
        # Items to hold:
        # Original data sources, original list of variables
        # Intersection between the two convex hulls, as polygon
        # Intersection between the two convex hulls, as list of data
        # List of all data not in intersection between both
        # Ability to check additional, future data (ability to update or not)

    def get_hull_points(self) -> Tuple[pd.DataFrame]:
        treated_hull = self.treatment_data.loc[self.treated_convex_hull.vertices]
        nontreated_hull = self.nontreatment_data.loc[self.nontreated_convex_hull.vertices]

        return treated_hull, nontreated_hull

    def get_convex_hull(self, data: pd.DataFrame) -> ConvexHull:
        covariates = data[self.list_of_variables]
        return ConvexHull(covariates)

    def get_hull_intersection(self) -> ConvexHull:
        # TODO: update to return convex hull of intersection.
        # If already done intersection, just return that.
        if self.intersection:
            return self.intersection
        # Create list of halfspaces
        treated_halfspaces = self.treated_convex_hull.equations
        nontreated_halfspaces = self.nontreated_convex_hull.equations

        total_halfspace = np.vstack(
            (treated_halfspaces, nontreated_halfspaces))

        self.valid_points, self.invalid_points = self.generate_list_of_valid_points(
            halfspace_equations=total_halfspace)

        interior_point = self.valid_points.iloc[0][self.list_of_variables]
        # Perform the intersection
        halfspace_intersection = HalfspaceIntersection(
            total_halfspace, interior_point=interior_point)

        self.intersection = ConvexHull(halfspace_intersection.intersections)

        # Return the intersection
        return self.intersection

    def check_point_for_positivity(self, test_point: np.array):

        pass

    def get_distance_point_to_hull(self, test_point: np.array, hull: ConvexHull) -> float:
        # Get points on simplices to check
        vertex_points = hull.points[hull.vertices]
        simplex_points = hull.points[hull.simplices]

        # Algorithm:
        # 1. Check point in comparison to all half spaces in dimension n
        tp = np.insert(test_point, len(self.list_of_variables), 1)
        distance_check = hull.equations @ tp.T
        halfspace_violations = np.sum(distance_check > 0)

        # If there are no halfspace violations, then the point is inside the hull.
        if halfspace_violations == 0:
            return 0

        dim = len(simplex_points.shape) - 1

        # Then, we start checking from dimension 1 items (points) to the nth dimension (hyperplanes)
        # If any lower dimension is greater than a higher dimension, we stop preemptively.
        print(vertex_points)
        distance_to_vertices = np.apply_along_axis(
            lambda x: np.linalg.norm(x - test_point), axis=1, arr=vertex_points)

        # The closest vertex is guaranteed to be in the answer somehow.
        minimum_vertex = np.argmin(distance_to_vertices)
        minimum_vertex_distance = np.min(distance_to_vertices)

        minimum_vertex_neighbors = hull.neighbors[minimum_vertex]

        # Now, loop through the remaining dimensions and see if any distances larger
        # Below is for the hyperplane
        mesh = np.meshgrid(minimum_vertex,
                           *[minimum_vertex_neighbors for _ in range(dim, 1, -1)])
        print(f"mesh: {mesh}")

        def find_results(x):
            print(f"x={x}")
            # TODO: change to pseudo-inverse, which allows for computing across dimensions
            # TODO: change to distance rather than printing the result.
            print(vertex_points[x].T)
            return np.linalg.solve(
                vertex_points[x].T,
                np.ones(dim)
            )

        w = np.apply_along_axis(find_results, axis=0, arr=mesh)

        # Compute distance of point to hyperplane using wx+b / ||w|| formula
        print(results)

    def generate_distances_of_invalid_points(self, metric: str = "Euclidean") -> pd.DataFrame:
        if not self.intersection:
            self.get_hull_intersection()

        invalid_treated = self.invalid_points[(
            self.invalid_points["Origin"] == "Treated")]
        invalid_untreated = self.invalid_points[(
            self.invalid_points["Origin"] == "Untreated")]

        if metric == "Euclidean":
            # Equation is abs(wx + b)/ ||w|| for the normal equations of each plane.
            invalid_treated = invalid_treated[self.list_of_variables]
            invalid_untreated = invalid_untreated[self.list_of_variables]

            invalid_treated.insert(
                len(self.list_of_variables),
                "Intercept", 1
            )
            invalid_untreated.insert(
                len(self.list_of_variables),
                "Intercept", 1
            )

            # TODO: need to change to use above function

            # Distances less than 0 are inside the halfspaces.
            invalid_treated_distances = invalid_treated @ self.nontreated_convex_hull.equations.T / \
                np.linalg.norm(self.nontreated_convex_hull.equations)
            # Internal function first masks all values < 0 with the maximum, and then finds the minimum
            # The minimum is the distance to a boundary.
            invalid_treated_distances = invalid_treated_distances.apply(
                lambda x: min(x.mask(lambda y: y < 0, other=max(x))), axis=1
            )

            invalid_untreated_distances = invalid_untreated @ self.treated_convex_hull.equations.T / \
                np.linalg.norm(self.treated_convex_hull.equations)

            invalid_untreated_distances = invalid_untreated_distances.apply(
                lambda x: min(x.mask(lambda y: y < 0, other=max(x))), axis=1
            )

            distances = pd.concat([
                invalid_treated_distances, invalid_untreated_distances
            ])

            return distances

        raise Exception(f"Metric: {metric} not defined yet.")

    def generate_list_of_valid_points(self, halfspace_equations: np.array = None) -> pd.DataFrame:
        if self.valid_points and self.invalid_points:
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
                return None

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
