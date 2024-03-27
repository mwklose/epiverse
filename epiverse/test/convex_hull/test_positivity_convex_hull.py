import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epiverse.utilities.data_generation.data_generator_polygon import DataGeneratorPolygon
from epiverse.convex_hull.positivity_by_convex_hull import PositivityConvexHull
from icecream import ic


class TestPositivityConvexHull(unittest.TestCase):

    def test_get_convex_hull(self):

        dgp = DataGeneratorPolygon(["C1", "C2"], [0, 0], [1, 0], [1, 1])
        data = dgp.generate_data(n=1000, seed=100)

        pch = PositivityConvexHull(data, data, ["C1", "C2"])

        assert len(pch.treated_convex_hull.vertices) >= 3
        assert np.allclose(0.5, pch.treated_convex_hull.volume, atol=5e-2)

    def test_get_distance_point_to_hull(self):
        dgp = DataGeneratorPolygon(["C1", "C2"], [0, 0], [1, 0], [1, 1])
        data = dgp.generate_data(n=60000, seed=100)

        pch = PositivityConvexHull(data, data, ["C1", "C2"])

        hulls = pch.get_hull_intersection()
        test_distance = pch.get_distance_point_to_hull(
            np.array([0, 1]), hulls)

        assert np.allclose(np.sqrt(2) / 2, test_distance[0], atol=5e-4)

        upper_trapezoid_data = DataGeneratorPolygon(
            ["C1", "C2"], [0, 2], [1, 0.5], [2, 0.5], [3, 2]).generate_data(n=100, seed=123)
        lower_trapezoid_data = DataGeneratorPolygon(
            ["C1", "C2"], [0, 0], [3, 0], [2, 1.5], [1, 1.5]).generate_data(n=100, seed=123)

        pch_trapezoid = PositivityConvexHull(
            upper_trapezoid_data, lower_trapezoid_data, ["C1", "C2"])

        wrong_distance = pch_trapezoid.get_distance_point_to_hull(
            np.array([1.40205097, 1.48760352]), pch_trapezoid.get_hull_intersection())

        ic(wrong_distance)
        assert False
        assert not wrong_distance[0] > 0.1

    def test_check_positivity_point(self):
        upper_trapezoid_data = DataGeneratorPolygon(
            ["C1", "C2"], [0, 2], [1, 0.5], [2, 0.5], [3, 2]).generate_data(n=100, seed=123)
        lower_trapezoid_data = DataGeneratorPolygon(
            ["C1", "C2"], [0, 0], [3, 0], [2, 1.5], [1, 1.5]).generate_data(n=100, seed=123)

        pch_trapezoid = PositivityConvexHull(
            upper_trapezoid_data, lower_trapezoid_data, ["C1", "C2"])

        assert not pch_trapezoid.check_point_for_positivity(np.array([0, 0]))

        assert not pch_trapezoid.check_point_for_positivity(np.array([4, 4]))

        distance = pch_trapezoid.get_distance_point_to_hull(
            np.array([2, 1]), pch_trapezoid.get_hull_intersection())

        assert pch_trapezoid.check_point_for_positivity(np.array([2, 1]))

    def test_generate_distances_of_invalid_points(self):
        # 1b. Do for partial nonpositivity
        upper_trapezoid_data = DataGeneratorPolygon(
            ["C1", "C2"], [0, 2], [1, 0.5], [2, 0.5], [3, 2]).generate_data(n=100, seed=123)
        lower_trapezoid_data = DataGeneratorPolygon(
            ["C1", "C2"], [0, 0], [3, 0], [2, 1.5], [1, 1.5]).generate_data(n=100, seed=123)

        pch_trapezoid = PositivityConvexHull(
            upper_trapezoid_data, lower_trapezoid_data, ["C1", "C2"])

        pch_trapezoid.get_distance_point_to_hull(
            np.array([1.40205097, 1.48760352]), pch_trapezoid.get_hull_intersection())

        known_bad = pch_trapezoid.get_distance_point_to_hull(
            np.array([2.072644, 0.682927]), pch_trapezoid.get_hull_intersection())

        valid, invalid = pch_trapezoid.generate_list_of_valid_points()

        assert invalid[invalid["Distance"].isnull()].empty
