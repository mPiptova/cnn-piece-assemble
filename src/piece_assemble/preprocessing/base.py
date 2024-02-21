from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from PIL.Image import Image

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image


class PieceExtractorBase(ABC):
    @abstractmethod
    def __call__(self, img: Image) -> tuple[Image, np.ndarray]:
        pass
