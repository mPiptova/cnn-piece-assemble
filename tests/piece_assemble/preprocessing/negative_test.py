import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from piece_assemble.preprocessing import NegativePieceExtractor


@pytest.fixture
def negative_extractor_resize() -> NegativePieceExtractor:
    return NegativePieceExtractor(max_image_size=100)


@pytest.fixture
def negative_extractor() -> NegativePieceExtractor:
    return NegativePieceExtractor()


@pytest.fixture
def original_piece_img() -> PILImage:
    return Image.open("tests/piece_assemble/preprocessing/data/star.png")


def test_negative_resize(
    original_piece_img: PILImage, negative_extractor_resize: NegativePieceExtractor
) -> None:
    piece_img, mask = negative_extractor_resize(original_piece_img)
    assert mask.shape[0] == piece_img.size[1] and mask.shape[1] == piece_img.size[0]
    assert {val for val in np.unique(mask)} == {0, 1}
    assert mask.shape == (88, 90)


def test_negative(
    original_piece_img: PILImage, negative_extractor: NegativePieceExtractor
) -> None:
    piece_img, mask = negative_extractor(original_piece_img)
    assert mask.shape[0] == piece_img.size[1] and mask.shape[1] == piece_img.size[0]
    assert {val for val in np.unique(mask)} == {0, 1}
    assert mask.shape == (88, 94)
