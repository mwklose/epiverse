import unittest
from epiverse.utilities.data_generation.data_generator_polygon import DataGeneratorPolygon
from epiverse.utilities.geometry.point import Point


class TestDataGeneratorPolygon(unittest.TestCase):

    def test_creation(self):
        triangle_data = DataGeneratorPolygon(["C1", "C2"],
                                             [0, 0], [0, 1], [1, 1])
        self.assertEqual(len(triangle_data.polygon.triangulation), 1)
        self.assertEqual(triangle_data.polygon.triangulation_areas, [0.5])
        self.assertEqual(triangle_data.polygon.triangulation_probability, [1])

    def test_generation(self):
        upper_triangle_data = DataGeneratorPolygon(["C1", "C2"],
                                                   [0, 0], [0, 1], [1, 1])
        my_data = upper_triangle_data.generate_data(n=1000, seed=722)

        self.assertTrue(all(my_data['C1'] < my_data['C2']))

        lower_triangle_data = DataGeneratorPolygon(["C1", "C2"],
                                                   [0, 0], [1, 0], [1, 1])

        my_data = lower_triangle_data.generate_data(n=1000, seed=722)

        self.assertTrue(all(my_data['C1'] > my_data['C2']))

        square_data = DataGeneratorPolygon(["C1", "C2"],
                                           [0, 0], [1, 0], [1, 1], [0, 1])

        my_data = square_data.generate_data(n=1000, seed=722)
        my_data = my_data.apply(lambda row: Point(
            [row["C1"], row["C2"]]), axis=1)
        my_data = my_data.apply(
            lambda row: square_data.polygon.point_in_polygon(row))

        self.assertTrue(any(my_data))
