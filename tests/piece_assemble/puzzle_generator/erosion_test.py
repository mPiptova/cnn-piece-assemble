import numpy as np

from piece_assemble.puzzle_generator.erosion import (
    _get_erosion_prob,
    apply_random_erosion,
    generate_noise_image,
)


def test_generate_noise_image() -> None:
    noise_img = generate_noise_image((100, 100))
    assert noise_img.shape == (100, 100)
    assert noise_img.min() == 0 and noise_img.max() == 1


def test_get_erosion_prob() -> None:
    mask = np.ones((5, 5), dtype=bool)

    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.5, 0.5, 0.5, 1.0],
            [1.0, 0.5, 0.0, 0.5, 1.0],
            [1.0, 0.5, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    erosion_prob = _get_erosion_prob(mask, width=2)
    print(erosion_prob)
    assert erosion_prob.shape == (5, 5)
    assert (erosion_prob == expected).all()


def test_apply_random_erosion() -> None:
    mask = np.ones((5, 5), dtype=bool)
    img = np.ones((5, 5))
    noise_img = generate_noise_image((50, 50))

    expected_mask = np.array(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [False, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, False, True],
        ]
    )

    rng = np.random.default_rng(42)

    eroded_mask, eroded_img = apply_random_erosion(mask, img, rng, noise_img, 2)

    print(eroded_mask)
    assert eroded_mask.shape == (5, 5)
    assert eroded_img.shape == (5, 5)
    assert (eroded_mask == expected_mask).all()
