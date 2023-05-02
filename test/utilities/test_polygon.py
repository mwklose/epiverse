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
        self.assertListEqual([0.5, 0.5], square.triangulation_areas)
        for t in square.triangulation_areas:
            self.assertAlmostEqual(t, 0.5)

        arrowhead = EpiPolygon([0, 0], [2, 1], [0, 2], [1, 1])
        self.assertEqual(len(arrowhead.triangulation), 2)
        self.assertListEqual([0.5, 0.5], arrowhead.triangulation_areas)

        hook = EpiPolygon([0, 0], [0, -1], [1, -1], [1, 1], [2, 0], [0.5, 2])
        self.assertEqual(len(hook.triangulation), 4)

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

    def test_point_operations(self):
        self.assertEqual(len(Point([1, 5])), 2)
        self.assertEqual(len(Point([1, 2, 3, 4])), 4)

        p1 = Point([4, 5])
        p2 = Point([2, 2])
        self.assertEqual(Point([2, 3]), p1-p2)

        self.assertEqual(Point([2, 3]) + p2, p1)

        self.assertEqual(Point([15, 10]) / 5, Point([3, 2]))
        self.assertEqual(Point([6, 4]) / Point([2, 2]), Point([3, 2]))
        self.assertEqual(Point([2, 5]) * 2, Point([4, 10]))
        self.assertEqual(Point([2, 5]) * Point([1, 2]), Point([2, 10]))
