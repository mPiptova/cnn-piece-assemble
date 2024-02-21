from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PIL.Image import Image
import numpy as np


if TYPE_CHECKING:
    from PIL.Image import Image
    import numpy as np


class PieceExtractorBase(ABC):
    @abstractmethod
    def __call__(self, img: Image) -> tuple[Image, np.ndarray]:
        pass
