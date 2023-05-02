from dataclasses import dataclass, field
from typing import List
from numbers import Number
import numpy as np


@dataclass
class Point:
    coords: List[float] = field(default_factory=list)

    def distance_to_origin(self) -> float:
        return sum([c**2 for c in coords])**0.5

    def ccw_array(self) -> np.array:
        return np.array([*self.coords, 1])

    def __len__(self):
        return len(self.coords)

    def __add__(self, other):
        if not isinstance(other, Point):
            raise TypeError(
                "Addition only defined for Point and Point.")

        if len(self) != len(other):
            raise Exception(
                "To add points, they must be in same dimension.")
        coord_sum = [e1 + e2 for e1,
                     e2 in zip(self.coords, other.coords)]
        return Point(coord_sum)

    def __sub__(self, other):
        if not isinstance(other, Point):
            raise TypeError(
                "Subtraction only defined for Point and Point.")

        if len(self) != len(other):
            raise Exception(
                "To subtract points, they must be in same dimension.")
        coord_difference = [e1 - e2 for e1,
                            e2 in zip(self.coords, other.coords)]
        return Point(coord_difference)

    def __mul__(self, other):
        if isinstance(other, Point) and len(self) == len(other):
            coord_mul = [e1 * e2 for e1, e2 in zip(self.coords, other.coords)]
            return Point(coords=coord_mul)

        coord_mul = [e1 * other for e1 in self.coords]
        return Point(coords=coord_mul)

    def __truediv__(self, other):
        if isinstance(other, Point) and len(self) == len(other):
            coord_div = [e1 / e2 for e1, e2 in zip(self.coords, other.coords)]
            return Point(coords=coord_div)

        coord_div = [e1 / other for e1 in self.coords]
        return Point(coords=coord_div)
