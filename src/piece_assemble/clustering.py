from __future__ import annotations

from functools import cached_property
from itertools import combinations

import cv2 as cv
import numpy as np
import shapely
from more_itertools import flatten
from scipy.ndimage import gaussian_filter1d
from shapely import Polygon
from shapely.ops import unary_union
from skimage.transform import rotate

from piece_assemble.geometry import (
    Transformation,
    get_common_contour,
    get_common_contour_idxs,
)
from piece_assemble.piece import Piece
from piece_assemble.types import Points
from piece_assemble.utils import longest_continuous_subsequence
from piece_assemble.visualization import draw_contour


class ClusterScorer:
    def __init__(
        self,
        w_convexity: float,
        w_complexity: float,
        w_color_dist: float,
        w_dist: float,
        w_hole_area: float,
        min_allowed_hole_size,
    ) -> None:
        self.w_convexity = w_convexity
        self.w_complexity = w_complexity
        self.w_color_dist = w_color_dist
        self.w_dist = w_dist
        self.w_hole_area = w_hole_area
        self.min_allowed_hole_size = min_allowed_hole_size

    def __call__(self, cluster: Cluster) -> float:
        convexity_score = cluster.convexity * self.w_convexity
        complexity_score = (
            cluster.complexity * cluster.avg_neighbor_count * self.w_complexity
        )
        color_score = -cluster.color_dist * self.w_color_dist
        dist_score = -cluster.dist * self.w_dist
        hole_score = (
            0
            if cluster.max_hole_area > self.min_allowed_hole_size
            else -cluster.max_hole_area * self.w_hole_size
        )
        return (
            convexity_score + complexity_score + color_score + dist_score + hole_score
        )


class MergeError(Exception):
    """Exception for errors during cluster merging."""

    pass


class DisjunctClustersError(MergeError):
    """Exception raised when two disjunct clusters are merged."""

    pass


class ConflictingTransformationsError(MergeError):
    """Exception raised when two clusters have conflicting transformations."""

    pass


class SelfIntersectionError(MergeError):
    """Exception raised when the intersection of the cluster is too high."""


class Cluster:
    def __init__(
        self,
        pieces: dict[str, tuple[Piece, Transformation]],
        scorer: ClusterScorer,
        parents: list[Cluster] = None,
    ) -> None:
        self._pieces = pieces

        self.descriptors = {key: desc for key, (desc, _) in pieces.items()}
        self.transformations = {key: t for key, (_, t) in pieces.items()}
        self.parents = parents

    @cached_property
    def score(self) -> float:
        return self.scorer(self)

    @cached_property
    def border(self) -> Points:
        border = []
        for key1, key2 in combinations(self.piece_ids, 2):
            b1, b2 = get_common_contour(
                self.transformations[key1].apply(self.descriptors[key1].contour),
                self.transformations[key2].apply(self.descriptors[key2].contour),
                4,
            )
            border.append((b1 + b2) / 2)

        return np.concatenate(border)

    @cached_property
    def border_length(self) -> int:
        return len(self.border)

    @property
    def piece_ids(self) -> set[str]:
        return set(self._pieces.keys())

    def copy(self) -> Cluster:
        new_cluster = Cluster(self._pieces.copy())
        new_cluster.border_length = self.border_length
        return new_cluster

    def transform(self, transformation: Transformation) -> Cluster:
        new_pieces = {
            key: (desc, t.compose(transformation))
            for key, (desc, t) in self._pieces.items()
        }
        new_cluster = Cluster(new_pieces)
        new_cluster.border_length = self.border_length
        return new_cluster

    @cached_property
    def dist(self) -> float:
        dists = []
        for key1, key2 in combinations(self.piece_ids, 2):
            b1, b2 = get_common_contour(
                self.transformations[key1].apply(self.descriptors[key1].contour),
                self.transformations[key2].apply(self.descriptors[key2].contour),
                4,
            )
            dists.append(np.linalg.norm(b1 - b2, axis=1))
        dists = np.concatenate(dists)
        return np.mean(dists)

    @cached_property
    def self_intersection(self) -> float:
        polygons = [
            shapely.transform(desc.polygon, lambda pol: t.apply(pol))
            for desc, t in self._pieces.values()
        ]
        return max(
            [
                p1.intersection(p2).area / min(p1.area, p2.area)
                for p1, p2 in combinations(polygons, 2)
            ]
        )

    @property
    def transformed_polygons(self) -> list[Polygon]:
        return [
            shapely.transform(desc.polygon, lambda pol: t.apply(pol))
            for desc, t in self._pieces.values()
        ]

    def intersection(self, polygon: Polygon) -> float:
        polygons = self.transformed_polygons
        return max(
            [p.intersection(polygon).area / min(p.area, polygon.area) for p in polygons]
        )

    @cached_property
    def polygon_union(self) -> Polygon:
        polygons = self.transformed_polygons
        polygons = [polygon.buffer(1) for polygon in polygons]
        return unary_union(polygons)

    @cached_property
    def max_hole_area(self):
        union = self.polygon_union

        if union.geom_type == "Polygon":
            polygons = [union]
        else:
            polygons = union.geoms
        hole_areas = [
            [Polygon(hole.coords).area for hole in polygon.interiors]
            for polygon in polygons
        ]
        hole_areas = list(flatten(hole_areas))
        if len(hole_areas) == 0:
            return 0
        return np.max(hole_areas)

    def merge(self, other: Cluster, self_intersection_tol=0.04) -> Cluster:
        common_keys = self.piece_ids.intersection(other.piece_ids)

        if len(common_keys) == 0:
            raise DisjunctClustersError(
                f"Pieces {self.piece_ids} and {other.piece_ids} "
                "have no common elements."
            )

        common_key = common_keys.pop()
        cluster1 = self.transform(self.transformations[common_key].inverse())
        cluster2 = other.transform(other.transformations[common_key].inverse())

        for key in common_keys:
            if not cluster1.transformations[key].is_close(
                cluster2.transformations[key]
            ):
                raise ConflictingTransformationsError(
                    f"Transformations {cluster1.transformations[key]} and "
                    f"{cluster2.transformations[key]} are not close."
                )

        new_pieces = cluster1._pieces.copy()
        new_pieces.update(cluster2._pieces)
        new_cluster = Cluster(new_pieces, parents=[cluster1, cluster2])

        if new_cluster.self_intersection > self_intersection_tol:
            raise SelfIntersectionError(
                f"Self intersection {new_cluster.self_intersection} "
                "is higher than tolerance {self_intersection_tol}"
            )

        return new_cluster

    @cached_property
    def convexity(self) -> float:
        union_polygon = self.polygon_union
        return union_polygon.area / union_polygon.convex_hull.area

    def indicator(self, all_ids):
        return np.array(
            [True if piece_id in self.piece_ids else False for piece_id in all_ids]
        )

    def get_match_complexity(self, key1: str, key2: str):
        piece1 = self.descriptors[key1]
        piece2 = self.descriptors[key2]
        transformation1 = self.transformations[key1]
        transformation2 = self.transformations[key2]

        _, idxs2 = get_common_contour_idxs(
            transformation1.apply(piece1.contour),
            transformation2.apply(piece2.contour),
            5,
        )
        if len(idxs2) == 0:
            return 0

        idxs2 = np.concatenate((idxs2, idxs2 + len(piece2.contour)))
        idxs2 = longest_continuous_subsequence(np.unique(idxs2))
        idxs2 = idxs2 % len(piece2.contour)

        if len(idxs2) == 0:
            return 0
        unique_arc_idxs1 = np.unique(
            piece2.contour_segment_idxs[idxs2], return_counts=True
        )
        arc_idxs2 = [
            idx
            for idx, count in zip(*unique_arc_idxs1)
            if idx != -1 and count > 0.8 * len(piece2.segments[idx])
        ]
        if len(arc_idxs2) <= 2:
            return 0
        return len(arc_idxs2)

    @cached_property
    def complexity(self):
        total_complexity = 0
        for key1, key2 in combinations(self.piece_ids, 2):
            total_complexity += self.get_match_complexity(key1, key2)

        return total_complexity

    def get_match_color_dist(self, key1: str, key2: str):
        piece1 = self.descriptors[key1]
        piece2 = self.descriptors[key2]
        border_idxs1, border_idxs2 = get_common_contour_idxs(
            self.transformations[key1].apply(piece1.contour),
            self.transformations[key2].apply(piece2.contour),
            5,
        )
        if len(border_idxs1) == 0:
            return -1
        border1 = piece1.contour[border_idxs1].round().astype(int)
        border2 = piece2.contour[border_idxs2].round().astype(int)

        values1 = piece1.img_avg[border1[:, 0], border1[:, 1]]
        values2 = piece2.img_avg[border2[:, 0], border2[:, 1]]

        values1 = gaussian_filter1d(values1, 10, axis=0)
        values2 = gaussian_filter1d(values2, 10, axis=0)

        values_diff = np.abs(values1 - values2)
        return np.mean(values_diff * values_diff)

    @cached_property
    def color_dist(self):
        dists = []
        for key1, key2 in combinations(self.piece_ids, 2):
            s = self.get_match_color_dist(key1, key2)
            if s != -1:
                dists.append(s)

        if len(dists) == 0:
            return 0.000001
        return np.max(dists)

    @cached_property
    def neighbor_matrix(self):
        piece_ids = list(self.piece_ids)
        matrix = np.full([len(self.descriptors)] * 2, False)
        for i1, i2 in combinations(range(len(piece_ids)), 2):
            complexity = self.get_match_complexity(piece_ids[i1], piece_ids[i2])
            if complexity == 0:
                continue
            matrix[i1, i2] = True
            matrix[i2, i1] = True

        return matrix

    @cached_property
    def avg_neighbor_count(self):
        return np.sum(self.neighbor_matrix, axis=0).mean()

    def draw(self, draw_contours: bool = False) -> np.ndarray:
        min_row, min_col, max_row, max_col = np.inf, np.inf, -np.inf, -np.inf

        piece_imgs = []
        center_positions = []
        for piece, transformation in self._pieces.values():
            deg_angle = np.rad2deg(transformation.rotation_angle)
            rot_img = rotate(
                np.where(piece.mask[:, :, np.newaxis], piece.img, -1),
                -deg_angle,
                resize=True,
                mode="constant",
                cval=-1,
            )

            # Crop the image symmetrically to keep the center position
            col, row, w, h = cv.boundingRect((rot_img[:, :, 0] != -1).astype("uint8"))
            row = min(row, rot_img.shape[0] - h - row)
            col = min(col, rot_img.shape[1] - w - col)
            h = rot_img.shape[0] - 2 * row
            w = rot_img.shape[1] - 2 * col
            rot_img = rot_img[
                row : rot_img.shape[0] - row, col : rot_img.shape[1] - col
            ]

            piece_imgs.append(rot_img)

            center_orig = (piece.contour.max(axis=0) + piece.contour.min(axis=0)) / 2
            center_target = transformation.apply(center_orig)
            center_positions.append(center_target.round().astype(int))

            min_row = min(min_row, center_target[0] - rot_img.shape[0] / 2)
            min_col = min(min_col, center_target[1] - rot_img.shape[1] / 2)
            max_row = max(max_row, center_target[0] + rot_img.shape[0] / 2)
            max_col = max(max_col, center_target[1] + rot_img.shape[1] / 2)

        offset = np.array((min_row, min_col))
        size = (int(round(max_row - min_row)), int(round(max_col - min_col)), 3)
        img = np.ones(size)

        for piece_img, center_pos in zip(piece_imgs, center_positions):
            top_left = np.maximum(
                0, center_pos - offset - (np.array(piece_img.shape[:2]) // 2)
            ).astype(int)
            img_crop = img[
                top_left[0] : top_left[0] + piece_img.shape[0],
                top_left[1] : top_left[1] + piece_img.shape[1],
            ]
            if img_crop.shape != piece_img.shape:
                piece_img = piece_img[: img_crop.shape[0], : img_crop.shape[1]]
            img[
                top_left[0] : top_left[0] + piece_img.shape[0],
                top_left[1] : top_left[1] + piece_img.shape[1],
            ] = np.where(piece_img < 0, img_crop, piece_img)

        if draw_contours:
            contours = [
                value[1].apply(value[0].contour) for value in self._pieces.values()
            ]
            contours = (np.concatenate(contours) - offset).round().astype(int)
            contours = contours[(contours[:, 0] < size[0]) & (contours[:, 1] < size[1])]
            img_contour = np.ones((size[0], size[1]))
            img_contour = draw_contour(contours, img_contour)
            img = np.where(
                img_contour[:, :, np.newaxis] == 0, np.array([[[1, 0, 0]]]), img
            )
        return img
