from __future__ import annotations

from functools import cached_property
from itertools import combinations

import cv2 as cv
import numpy as np
import shapely
from shapely import Polygon
from shapely.ops import unary_union
from skimage.transform import rotate

from piece_assemble.geometry import Transformation, get_common_contour_length
from piece_assemble.piece import Piece
from piece_assemble.visualization import draw_contour


class Cluster:
    def __init__(self, pieces: dict[str, tuple[Piece, Transformation]]) -> None:
        self._pieces = pieces

        self.descriptors = {key: desc for key, (desc, _) in pieces.items()}
        self.transformations = {key: t for key, (_, t) in pieces.items()}

    @cached_property
    def border_length(self) -> int:
        total_length = 0
        for key1, key2 in combinations(self.piece_ids, 2):
            total_length += get_common_contour_length(
                self.transformations[key1].apply(self.descriptors[key1].contour),
                self.transformations[key2].apply(self.descriptors[key2].contour),
                4,
            )
        return total_length

    @property
    def piece_ids(self) -> set[str]:
        return set(self._pieces.keys())

    def copy(self) -> Cluster:
        new_cluster = Cluster(self._pieces.copy())
        new_cluster.border_length = self.border_length
        return new_cluster

    def add(self, descriptor: Piece, transformation: Transformation) -> None:
        if descriptor.name in self.piece_ids:
            # TODO: Create more meaningful error
            raise ValueError()

        self._pieces[descriptor.name] = (descriptor, transformation)
        self.descriptors[descriptor.name] = descriptor
        self.transformations[descriptor.name] = transformation

        for key in self.piece_ids:
            if key == descriptor.name:
                continue
            self.border_length += get_common_contour_length(
                descriptor.contour, self.descriptors[key].contour
            )

        # TODO: Update self_intersection and score

    def transform(self, transformation: Transformation) -> Cluster:
        new_pieces = {
            key: (desc, t.compose(transformation))
            for key, (desc, t) in self._pieces.items()
        }
        new_cluster = Cluster(new_pieces)
        new_cluster.border_length = self.border_length
        return new_cluster

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

    def merge(self, other: Cluster) -> Cluster:
        common_keys = self.piece_ids.intersection(other.piece_ids)

        if len(common_keys) == 0:
            # TODO: more meaningful error
            raise ValueError

        common_key = common_keys.pop()
        cluster1 = self.transform(self.transformations[common_key].inverse())
        cluster2 = other.transform(other.transformations[common_key].inverse())

        for key in common_keys:
            if not cluster1.transformations[key].is_close(
                cluster2.transformations[key]
            ):
                # TODO: more meaningful error
                raise ValueError()

        new_pieces = cluster1._pieces.copy()
        new_pieces.update(cluster2._pieces)
        new_cluster = Cluster(new_pieces)

        return new_cluster

    @cached_property
    def score(self) -> float:
        # TODO: More sensible score
        return self.border_length * (self.convexity**5)

    @cached_property
    def convexity(self) -> float:
        polygons = self.transformed_polygons
        union_polygon = unary_union(polygons)
        return union_polygon.area / union_polygon.convex_hull.area

    def indicator(self, all_ids):
        return np.array(
            [True if piece_id in self.piece_ids else False for piece_id in all_ids]
        )

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
