import pytest

from piece_assemble.preprocessing.common import get_resize_shape


@pytest.mark.parametrize(
    "original_shape, max_size, expected_new_shape",
    [
        ((100, 100), 100, (100, 100)),
        ((100, 200), 100, (50, 100)),
        ((200, 100), 100, (100, 50)),
        ((200, 200), 100, (100, 100)),
        ((400, 200), 100, (100, 50)),
        ((200, 400), 100, (50, 100)),
    ],
)
def test_get_resize_shape(
    original_shape: tuple[int, int], max_size: int, expected_new_shape: tuple[int, int]
) -> None:
    new_shape = get_resize_shape(original_shape, max_size)
    assert new_shape == expected_new_shape
