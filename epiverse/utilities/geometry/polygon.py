
from typing import List
from epiverse.utilities.geometry.point import Point
import itertools
import numpy as np


class EpiPolygon:
    def __init__(self, *args):
        self.points = self.check_points(args)
        self.triangulation = self.triangulate(
            self.points)
        self.triangulation_areas = self.area_of_triangulations()
        self.triangulation_probability = self.triangulation_areas / \
            sum(self.triangulation_areas)

    def check_points(self, points: List) -> List:
        if not points:
            raise Exception("List of points in EpiPolygon must be non-zero.")

        initial_dimension = len(points[0])

        for pt in points:
            # Check that all points have same number of dimensions
            if len(pt) != initial_dimension:
                raise Exception(
                    f"Point {pt} does not have {initial_dimension} coordinates")

        return [Point(pt) for pt in points if not isinstance(pt, Point)]

    # TODO: handle issues of collinearity.
    # TODO: generalize to multiple dimensions
    def triangulate(self, points: List) -> List[List[Point]]:
        # Base case: 3 points left in list of points, is definitely a triangle.
        if len(points) == len(points[0]) + 1:
            return [points]

        # Loop through the previous 3-pairs of points.
        for i, pt in enumerate(points):
            ccw = EpiPolygon.is_ccw(
                [points[i - 2], points[i - 1], pt])
            # Case 1: Next 3 points form CCW (is convex).
            if ccw > 0:
                # Case 1b: Other points exist within the triangle. Greedy approach to see whether those points are better suited.
                current_points = [points[i-2], points[i-1], points[i]]
                remaining_points = points[i+1: i-2]
                within_triangle = [
                    EpiPolygon.point_in_triangle(current_points, remaining) for remaining in remaining_points
                ]
                if within_triangle.count(False) == len(within_triangle):
                    # Case 1a: No other points exist within the triangle. Return this simplex, plus another one
                    break
            if ccw == 0:
                # TODO: case when 3 points are collinear, remove middle point (as no longer needed)
                pass
        else:
            print("All CW; Reversing list of points")
            self.triangulate(reverse(points))

        # Index i-1 no longer visible.
        other_points_to_triangulate = [
            points[i]] + remaining_points + [points[i-2]]

        other_triangulations = self.triangulate(other_points_to_triangulate)
        return [current_points] + other_triangulations

    def point_in_triangle(points: List, test_point: Point) -> bool:

        point_groups = [[points[i-1], pt, test_point]
                        for i, pt in enumerate(points)]
        point_groups_ccw = [EpiPolygon.is_ccw(group) for group in point_groups]

        inside_triangle = point_groups_ccw.count(
            point_groups_ccw[0]) == len(point_groups_ccw)

        return inside_triangle

    def point_in_polygon(self, test_point: Point) -> bool:
        for triangle in self.triangulation:
            if EpiPolygon.point_in_triangle(triangle, test_point):
                return True
        else:
            return False

    def is_ccw(points: List):
        ccw_arrays = [pt.ccw_array() for pt in points]
        ccw_array = np.array(ccw_arrays)
        det = np.linalg.det(ccw_array)

        return np.sign(det)

    def area_of_triangulations(self) -> List[float]:
        areas = []
        for triangle in self.triangulation:
            s1 = triangle[2] - triangle[0]
            s2 = triangle[1] - triangle[0]

            det = np.linalg.det(np.array([s1.coords, s2.coords])) / 2

            areas.append(np.absolute(det))
        return areas
