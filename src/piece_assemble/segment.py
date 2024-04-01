import numpy as np

from piece_assemble.geometry import extend_interval
from piece_assemble.types import Interval, Point, Points


class Segment:
    def __init__(self, interval: Interval, contour: Points):
        self.interval = interval
        ex_interval = extend_interval(interval, len(contour))
        idxs = np.arange(ex_interval[0], ex_interval[1]) % len(contour)
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
