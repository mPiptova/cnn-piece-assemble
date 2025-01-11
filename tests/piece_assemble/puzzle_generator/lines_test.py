import numpy as np

from piece_assemble.puzzle_generator.lines import (
    draw_curve,
    generate_random_line,
    get_random_point_on_side,
    get_random_points_on_different_sides,
    interpolate_curve,
    perturbate_points,
    sample_points_on_line,
)


def test_get_random_points_on_side() -> None:
    height, width = 100, 200
    rng = np.random.default_rng(42)

    points = [get_random_point_on_side("left", height, width, rng) for _ in range(10)]
    assert all(point[1] == 0 for point in points)

    points = [get_random_point_on_side("right", height, width, rng) for _ in range(10)]
    assert all(point[1] == width - 1 for point in points)

    points = [get_random_point_on_side("top", height, width, rng) for _ in range(10)]
    assert all(point[0] == 0 for point in points)

    points = [get_random_point_on_side("bottom", height, width, rng) for _ in range(10)]
    assert all(point[0] == height - 1 for point in points)


def test_get_random_points_on_different_sides() -> None:
    height, width = 100, 200
    rng = np.random.default_rng(42)

    points = get_random_points_on_different_sides(height, width, rng)
    assert points[0] == (43, 0)
    assert points[1] == (99, 86)


def test_generate_random_line() -> None:
    height, width = 100, 200
    rng = np.random.default_rng(42)
    points = generate_random_line(height, width, rng)
    assert points == ((43, 0), (99, 86))


def test_sample_points_on_line() -> None:
    height, width = 100, 200
    rng = np.random.default_rng(42)
    points = generate_random_line(height, width, rng)
    for _ in range(10):
        sampled_points = sample_points_on_line(points[0], points[1], 10, rng)
        assert sampled_points.shape == (10, 2)
        assert (min(points[0][0], points[1][0]) <= sampled_points[:, 0]).all()
        assert (sampled_points[:, 0] <= max(points[0][0], points[1][0])).all()
        assert (min(points[0][1], points[1][1]) <= sampled_points[:, 1]).all()
        assert (sampled_points[:, 1] <= max(points[0][1], points[1][1])).all()


def test_perturbate_points() -> None:
    height, width = 100, 200
    rng = np.random.default_rng(42)
    points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    perturbed_points = perturbate_points(points, 1, height, width, rng).astype(int)

    target = np.array([[0, 0], [1, 1], [0, 0], [3, 2], [3, 3], [5, 5]])

    assert (perturbed_points == target).all()


def test_interpolate_curve() -> None:
    points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    for n in [10, 50, 100, 500]:
        curve = interpolate_curve(points, n)
        assert curve.shape == (n, 2)


def test_draw_curve() -> None:
    points = np.array([[0, 0], [100, 3], [10, 200]])
    curve = interpolate_curve(points, 100)
    img = draw_curve(curve, 100, 200, 1)
    assert img.shape == (100, 200)
    assert (np.unique(img) == [0, 1]).all()
