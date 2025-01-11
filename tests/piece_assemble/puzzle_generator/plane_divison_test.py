import numpy as np
import pytest

from piece_assemble.geometry import Transformation
from piece_assemble.puzzle_generator.lines import interpolate_curve
from piece_assemble.puzzle_generator.plane_division import (
    add_division_by_curve,
    add_division_level,
    apply_division_to_image,
    crop_piece_img,
    divide_plane_by_curve,
    get_puzzle_division,
    get_random_division,
    reduce_number_of_pieces,
)


def test_divide_plane_by_curve() -> None:
    rng = np.random.default_rng(42)
    # split image in half by straight line
    points = np.array([[0, 5], [10, 5]])
    curve = interpolate_curve(points, 10)
    width, height = 10, 10

    img = divide_plane_by_curve(curve, height, width, rng)
    assert img.shape == (height, width)

    assert (img[:, :7] == 0).all()
    assert (img[:, 7:] == 1).all()


def test_add_division_by_curve() -> None:
    rng = np.random.default_rng(42)

    curve_vertical = interpolate_curve(np.array([[0, 5], [10, 5]]), 10)
    curve_horizontal = interpolate_curve(np.array([[5, 0], [5, 10]]), 10)
    width, height = 10, 10

    img = divide_plane_by_curve(curve_vertical, height, width, rng)
    img = add_division_by_curve(img, curve_horizontal, rng)
    assert img.shape == (height, width)

    assert (img[:7, :7] == 0).all()
    assert (img[:7, 7:] == 1).all()


def test_add_division_level() -> None:
    rng = np.random.default_rng(42)

    for size in [10, 100, 500]:
        img = np.zeros((size, size))
        for _ in range(3):
            img = add_division_level(img, 10, 1, rng)
            assert img.shape == (size, size)
            assert (img <= 1).all()
            assert (img >= 0).all()


def test_get_random_division() -> None:
    rng = np.random.default_rng(42)

    for size in [100, 500]:
        img = get_random_division(size, size, 5, 10, rng)
        assert img.shape == (size, size)
        assert (img <= 1).all()
        assert (img >= 0).all()


@pytest.mark.parametrize(
    "size, num_pieces, min_piece_area, result",
    [
        (
            4,
            2,
            1,
            np.array(
                [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 16],
                ]
            ),
        ),
        (
            4,
            4,
            1,
            np.array(
                [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 14, 15, 16],
                ]
            ),
        ),
        (
            4,
            2,
            2,
            np.array(
                [
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                ]
            ),
        ),
    ],
)
def test_reduce_number_of_pieces(
    size: int, num_pieces: int, min_piece_area: int, result: np.ndarray
) -> None:
    img = (np.arange(size * size) / size * size).reshape(size, size)

    reduced_img = reduce_number_of_pieces(img, num_pieces, min_piece_area)
    print(reduced_img)

    assert reduced_img.shape == (size, size)
    assert (reduced_img == result).all()


def test_get_puzzle_division() -> None:
    rng = np.random.default_rng(42)

    for i in range(5, 10):
        img = get_puzzle_division(100, 100, i, 1, 5, 10, rng)
        assert img.shape == (100, 100)
        assert (np.unique(img) == np.arange(1, i + 1)).all()


@pytest.mark.parametrize(
    "transformation, expected_transformation",
    [
        (Transformation.identity(), Transformation(0, np.array([-3, -3]))),
        (Transformation(0, np.array([0, 1])), Transformation(0, np.array([-3, -2]))),
        (
            Transformation(np.pi / 2, np.array([0, 0])),
            Transformation(np.pi / 2, np.array([-3, -3])),
        ),
        (
            Transformation(np.pi / 3, np.array([0, 0])),
            Transformation(np.pi / 3, np.array([-3, -3])),
        ),
        (
            Transformation(np.pi / 3, np.array([1, 1])),
            Transformation(np.pi / 3, np.array([-2, -2])),
        ),
    ],
)
def test_crop_piece_img(
    transformation: Transformation, expected_transformation: Transformation
) -> None:
    img = np.zeros((10, 10))
    img[3:7, 3:7] = 1

    cropped_img, mask, crop_transformation = crop_piece_img(
        img, img.astype(bool), transformation
    )
    assert cropped_img.shape == (4, 4)
    assert (cropped_img == 1).all()
    assert mask.all()
    assert crop_transformation.is_close(expected_transformation, 0.001, 0.001)


def test_apply_division_to_image() -> None:
    rng = np.random.default_rng(42)

    division = np.arange(5 * 5).reshape(5, 5) + 1
    division = np.repeat(division, 4, axis=0)
    division = np.repeat(division, 4, axis=1)
    print(division)
    print(division.shape)
    img = np.zeros((20, 20))

    pieces = apply_division_to_image(img, division, rng)
    assert len(pieces) == 5 * 5
