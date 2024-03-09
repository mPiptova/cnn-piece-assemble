import numpy as np
from PIL.Image import Image

Point = np.ndarray[float]
Points = np.ndarray[float]

# Image
BinImg = np.ndarray[bool]
NpImage = np.ndarray[float]
PilImage = Image

Interval = tuple[int, int] | tuple[float, float]
