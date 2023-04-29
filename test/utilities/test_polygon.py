import unittest
from src.utilities.geometry.point import Point
from src.utilities.geometry.polygon import EpiPolygon


class TestPolygon(unittest.TestCase):

    def test_creation(self):
        with self.assertRaises(Exception) as ar:
            EpiPolygon()
            self.assertEquals(str(ar.exception),
                              "List of points in EpiPolygon must be non-zero.")

        square = EpiPolygon([0, 0], [1, 0], [1, 1], [0, 1])
        self.assertEqual(len(square.triangulation), 2)

        arrowhead = EpiPolygon([0, 0], [2, 1], [0, 2], [1, 1])
        print(arrowhead.triangulation)
        self.assertEqual(len(arrowhead.triangulation), 2)

    def test_is_ccw(self):

        coords_ccw = [[1, 0], [2, 1], [1, 2]]
        points_ccw = [Point(coord) for coord in coords_ccw]

        self.assertTrue(EpiPolygon.is_ccw(points_ccw) > 0)

        coords_cw = [[1, 0], [1, 2], [2, 1]]
        points_cw = [Point(coord) for coord in coords_cw]

        self.assertTrue(EpiPolygon.is_ccw(points_cw) < 0)

        coords_linear = [[1, 0], [2, 1], [3, 2]]
        points_linear = [Point(coord) for coord in coords_linear]

        self.assertTrue(EpiPolygon.is_ccw(points_linear) == 0)
