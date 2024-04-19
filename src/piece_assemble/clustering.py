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

from piece_assemble.geometry import Transformation, get_common_contour_idxs, icp
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
        min_allowed_hole_size: float,
        w_border_length: float,
    ) -> None:
        self.w_convexity = w_convexity
        self.w_complexity = w_complexity
        self.w_color_dist = w_color_dist
        self.w_dist = w_dist
        self.w_hole_area = w_hole_area
        self.min_allowed_hole_size = min_allowed_hole_size
        self.w_border_length = w_border_length

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
            else -cluster.max_hole_area * self.w_hole_area
        )
        border_length_score = cluster.border_length * self.w_border_length
        return (
            convexity_score
            + complexity_score
            + color_score
            + dist_score
            + hole_score
            + border_length_score
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


class TransformedPiece:
    def __init__(self, piece: Piece, transformation: Transformation) -> None:
        self.piece = piece
        self.transformation = transformation

    def transform(self, transformation: Transformation) -> TransformedPiece:
        return TransformedPiece(self.piece, self.transformation.compose(transformation))


class Cluster:
    def __init__(
        self,
        pieces: dict[str, TransformedPiece],
        scorer: ClusterScorer,
        parents: list[Cluster] = None,
        self_intersection_tol: float = 0.01,
        border_dist_tol: float = 2,
    ) -> None:
        self.pieces = pieces
        self.parents = parents
        self.scorer = scorer
        self.self_intersection_tol = self_intersection_tol
        self.border_dist_tol = border_dist_tol

    @cached_property
    def score(self) -> float:
        return self.scorer(self)

    @cached_property
    def border(self) -> Points:
        border = []
        for (key1, key2) in self.matches_border_idxs.keys():
            b1, b2 = self.get_match_border_coordinates(key1, key2)
            if b1 is None:
                continue
            border.append((b1 + b2) / 2)

        if len(border) == 0:
            return []
        return np.concatenate(border)

    @cached_property
    def border_length(self) -> int:
        return len(self.border)

    @property
    def piece_ids(self) -> set[str]:
        return set(self.pieces.keys())

    def copy(self) -> Cluster:
        new_cluster = Cluster(self.pieces.copy(), self.scorer, self.parents)
        new_cluster.border_length = self.border_length
        return new_cluster

    def transform(self, transformation: Transformation) -> Cluster:
        new_pieces = {
            key: piece.transform(transformation) for key, piece in self.pieces.items()
        }
        new_cluster = Cluster(new_pieces, self.scorer, parents=[self])
        new_cluster.border_length = self.border_length
        return new_cluster

    @cached_property
    def dist(self) -> float:
        dists = []
        for key1, key2 in combinations(self.piece_ids, 2):
            b1, b2 = self.get_match_border_coordinates(key1, key2)
            if b1 is None:
                continue
            dists.append(np.linalg.norm(b1 - b2, axis=1))

        if len(dists) == 0:
            return np.inf

        dists = np.concatenate(dists)
        return np.mean(dists)

    @cached_property
    def self_intersection(self) -> float:
        polygons = self.transformed_polygons
        return max(
            [
                p1.intersection(p2).area / min(p1.area, p2.area)
                for p1, p2 in combinations(polygons, 2)
            ]
        )

    @property
    def transformed_polygons(self) -> list[Polygon]:
        return [
            shapely.transform(
                piece.piece.polygon, lambda pol: piece.transformation.apply(pol)
            )
            for piece in self.pieces.values()
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

    def merge(self, other: Cluster, finetune_iters: int = 3) -> Cluster:
        """Merge this cluster with another cluster.

        Parameters
        ----------
        other
            Cluster to be merged with.
        finetune_iters
            Number of finetuning iterations,
            After merging, ICP is run on all clusters to prevent cumulation
            of small errors.

        Returns
        -------
        New merged cluster.

        """
        common_keys = self.piece_ids.intersection(other.piece_ids)

        if len(common_keys) == 0:
            raise DisjunctClustersError(
                f"Pieces {self.piece_ids} and {other.piece_ids} "
                "have no common elements."
            )

        common_key = common_keys.pop()
        cluster1 = self.transform(self.pieces[common_key].transformation.inverse())
        cluster2 = other.transform(other.pieces[common_key].transformation.inverse())

        for key in common_keys:
            if not cluster1.pieces[key].transformation.is_close(
                cluster2.pieces[key].transformation
            ):
                raise ConflictingTransformationsError(
                    f"Transformations {cluster1.pieces[key].transformation} and "
                    f"{cluster2.piece[key].transformation} are not close."
                )

        new_pieces = cluster1.pieces.copy()
        new_pieces.update(cluster2.pieces)
        new_cluster = Cluster(
            new_pieces, parents=[cluster1, cluster2], scorer=self.scorer
        )

        if finetune_iters > 0:
            new_cluster = new_cluster.finetune_transformations(3)

        if new_cluster.self_intersection > self.self_intersection_tol:
            raise SelfIntersectionError(
                f"Self intersection {new_cluster.self_intersection} "
                "is higher than tolerance {self_intersection_tol}"
            )

        return new_cluster

    def can_be_merged(self, other: Cluster) -> Cluster:
        """Checks whether two clusters can be merged.

        Returns True if they share at least one piece and the relative position of
        the shared pieces is the same (within some tolerance).

        Parameters
        ----------
        other

        Returns
        -------
        Whether this cluster can be merged with the other cluster.
        """

        common_keys = self.piece_ids.intersection(other.piece_ids)
        if len(common_keys) == 0:
            return False

        common_key = common_keys.pop()
        cluster1 = self.transform(self.pieces[common_key].transformation.inverse())
        cluster2 = other.transform(other.pieces[common_key].transformation.inverse())

        return all(
            cluster1.pieces[key].transformation.is_close(
                cluster2.pieces[key].transformation
            )
            for key in common_keys
        )

    def finetune_transformations(self, num_iters: int = 3):
        """Improve transformations using ICP algorithm.

        Helps preventing cumulation of small errors in large clusters.

        Parameters
        ----------
        num_iters
            Number of iterations. Defines how many times a position of each cluster
            will be adjusted.

        Returns
        -------
        New finetuned cluster.
        """
        contour_dict = {
            key: piece.transformation.apply(piece.piece.contour)
            for key, piece in self.pieces.items()
        }

        new_pieces = self.pieces.copy()
        for _ in range(num_iters):
            for piece_id in self.piece_ids:
                piece_contour = contour_dict[piece_id]
                other_contours = np.concatenate(
                    [
                        contour
                        for _id, contour in contour_dict.items()
                        if _id != piece_id
                    ]
                )

                new_transform = icp(
                    piece_contour,
                    other_contours,
                    Transformation.identity(),
                    self.border_dist_tol,
                )
                new_pieces[piece_id] = new_pieces[piece_id].transform(new_transform)

                new_contour = new_transform.apply(piece_contour)
                contour_dict[piece_id] = new_contour

        return Cluster(new_pieces, self.scorer, self.parents)

    @cached_property
    def convexity(self) -> float:
        union_polygon = self.polygon_union
        return union_polygon.area / union_polygon.convex_hull.area

    def indicator(self, all_ids):
        return np.array(
            [True if piece_id in self.piece_ids else False for piece_id in all_ids]
        )

    @cached_property
    def matches_border_idxs(self):
        matches_border_dict = {}

        if self.parents is not None:
            for parent in self.parents:
                matches_border_dict.update(parent.matches_border_idxs)

        for key1, key2 in combinations(self.piece_ids, 2):

            if (key1, key2) in matches_border_dict.keys() or (
                key2,
                key1,
            ) in matches_border_dict.keys():
                continue
            piece1 = self.pieces[key1].piece
            piece2 = self.pieces[key2].piece
            transformation1 = self.pieces[key1].transformation
            transformation2 = self.pieces[key2].transformation
            idxs1, idxs2 = get_common_contour_idxs(
                transformation1.apply(piece1.contour),
                transformation2.apply(piece2.contour),
                self.border_dist_tol,
            )

            if len(idxs1) == 0:
                continue
            matches_border_dict[(key1, key2)] = (idxs1, idxs2)

        return matches_border_dict

    def get_match_border_idxs(self, key1, key2):
        if (key1, key2) in self.matches_border_idxs.keys():
            idxs1, idxs2 = self.matches_border_idxs[(key1, key2)]
        elif (key2, key1) in self.matches_border_idxs.keys():
            idxs2, idxs1 = self.matches_border_idxs[(key2, key1)]
        else:
            return None, None

        return idxs1, idxs2

    def get_match_border_coordinates(self, key1, key2):
        idxs1, idxs2 = self.get_match_border_idxs(key1, key2)
        if idxs1 is None:
            return None, None

        piece1 = self.pieces[key1].piece
        piece2 = self.pieces[key2].piece

        coords1 = self.pieces[key1].transformation.apply(piece1.contour[idxs1])
        coords2 = self.pieces[key2].transformation.apply(piece2.contour[idxs2])

        return coords1, coords2

    def get_match_complexity(self, key1: str, key2: str):
        _, idxs2 = self.get_match_border_idxs(key1, key2)
        if idxs2 is None:
            return 0

        piece2 = self.pieces[key2].piece

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
            if idx != -1 and count > 0.7 * len(piece2.segments[idx])
        ]
        if len(arc_idxs2) <= 1:
            return 0
        return len(arc_idxs2)

    @cached_property
    def complexity(self):
        total_complexity = 0
        for key1, key2 in combinations(self.piece_ids, 2):
            total_complexity += self.get_match_complexity(key1, key2)

        return total_complexity

    def get_match_color_dist(self, key1: str, key2: str):
        piece1 = self.pieces[key1].piece
        piece2 = self.pieces[key2].piece

        border_idxs1, border_idxs2 = self.get_match_border_idxs(key1, key2)
        if border_idxs1 is None:
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
        matrix = np.full([len(self.pieces)] * 2, False)
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
        for piece in self.pieces.values():
            transformation = piece.transformation
            piece = piece.piece
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
                value[1].apply(value[0].contour) for value in self.pieces.values()
            ]
            contours = (np.concatenate(contours) - offset).round().astype(int)
            contours = contours[(contours[:, 0] < size[0]) & (contours[:, 1] < size[1])]
            img_contour = np.ones((size[0], size[1]))
            img_contour = draw_contour(contours, img_contour)
            img = np.where(
                img_contour[:, :, np.newaxis] == 0, np.array([[[1, 0, 0]]]), img
            )
        return img
