import unittest

from epiverse.utilities.data_generation.data_generator_polygon import DataGeneratorPolygon
from epiverse.convex_hull.positivity_by_convex_hull import PositivityConvexHull


class TestPositivityConvexHull(unittest.TestCase):

    def test_get_convex_hull(self):

        dgp = DataGeneratorPolygon(["C1", "C2"], [0, 0], [1, 0], [1, 1])
        data = dgp.generate_data(n=1000, seed=100)

        pch = PositivityConvexHull(data, data, ["C1", "C2"])

        self.assertTrue(len(pch.treated_convex_hull.vertices) >= 3)
        self.assertAlmostEqual(0.5, pch.treated_convex_hull.volume, places=1)

    def test_get_hull_intersection(self):
        self.assertFalse(True)

    def test_check_positivity_point(self):
        self.assertFalse(True)

    def test_generate_list_of_valid_points(self):
        self.assertFalse(True)
