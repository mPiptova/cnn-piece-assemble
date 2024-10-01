import numpy as np

from geometry import extend_interval
from piece_assemble.types import Interval, Point, Points


class Segment:
    def __init__(self, interval: Interval, contour: Points, offset: int = 0):
        self.interval = interval
        self.offset = offset
        ex_interval = extend_interval(interval, len(contour))
        idxs = np.arange(ex_interval[0], ex_interval[1]) % len(contour) + offset
        self.contour = contour[idxs]

    def __len__(self) -> int:
        return len(self.contour)


class ApproximatingArc(Segment):
    def __init__(
        self,
        interval: Interval,
        contour: Points,
        center: Point,
        radius: float,
        contour_index: int,
    ):
        self.center = center
        self.radius = radius
        self.contour_index = contour_index
        super().__init__(interval, contour)
