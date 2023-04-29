from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Point:
    coords: List[float] = field(default_factory=list)

    def distance_to_origin(self) -> float:
        return sum([c**2 for c in coords])**0.5

    def ccw_array(self) -> np.array:
        return np.array([*self.coords, 1])
