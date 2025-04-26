from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from itertools import combinations
from typing import TYPE_CHECKING

import cv2 as cv
import numpy as np
from rustworkx import PyGraph, connected_components
from shapely import Polygon
from shapely.ops import unary_union
from skimage.morphology import disk, erosion
from skimage.transform import rotate

from piece_assemble.geometry import Transformation, get_common_contour_idxs, icp
from piece_assemble.neighbors import BorderLengthNeighborClassifier
from piece_assemble.piece import TransformedPiece
from piece_assemble.visualization import draw_contour

if TYPE_CHECKING:

    from piece_assemble.models.predict import Embeddings
    from piece_assemble.neighbors import NeighborClassifierBase
    from piece_assemble.types import NpImage, Points


class ClusterScorerBase(ABC):
    @abstractmethod
    def __call__(self, cluster: Cluster) -> float:

        pass


class DummyClusterScorer(ClusterScorerBase):
    def __call__(self, cluster: Cluster) -> float:
        return 0


class EmbeddingClusterScorer(ClusterScorerBase):
    """Computes the score of a cluster based on the similarity of the contour embeddings

    Weighted geometric mean of the embeddings
    """

    def __init__(
        self,
        embeddings: Embeddings,
        length_weight: float = 0.2,
    ) -> None:
        """
        Parameters
        ----------
        embeddings
            The embeddings to use for computing the score.
        length_weight
            The weight to give to the length of the contour.
        """
        super().__init__()
        self.embeddings = embeddings
        self.length_weight = length_weight

    def __call__(self, cluster: Cluster) -> float:
        score = 0
        for neighbor_pair in cluster.get_neighbor_pairs():
            neighbor_pair = list(neighbor_pair)
            matrix = self.embeddings.get_similarity_matrix(*neighbor_pair)

            idxs1, idxs2 = cluster.get_match_border_idxs(*neighbor_pair)
            if idxs1 is None or idxs2 is None:
                continue
            score += (matrix[idxs1, idxs2].sum() / len(idxs1)) * (
                len(idxs1) ** self.length_weight
            )

        cluster_score: float = score / (len(cluster.pieces) ** 0.5)
        return cluster_score


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
    """Class representing a cluster of pieces.

    Can be viewed as a partial puzzle solution.
    """

    def __init__(
        self,
        pieces: dict[str, TransformedPiece],
        scorer: ClusterScorerBase,
        self_intersection_tol: float,
        border_dist_tol: float,
        rotation_tol: float,
        translation_tol: float,
        neighbor_classifier: NeighborClassifierBase,
        parents: list[Cluster] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        pieces
            The pieces that make up the cluster, with their respective
            transformations.
        scorer
            The scorer to use for computing the score of the cluster.
        self_intersection_tol
            The tolerance for self intersection of the pieces, relative
            to their size.
        border_dist_tol
            The tolerance for piece borders to be considered as neighboring.
        rotation_tol
            The tolerance for rotation of the pieces
        translation_tol
            The tolerance for translation of the cluster.
        neighbor_classifier
            Defines how to determine if two pieces of the cluster are neighbors.
        parents
            Clusters from which this cluster was derived.
        """

        self.pieces = pieces
        self.parents = parents
        self.scorer = scorer
        self.self_intersection_tol = self_intersection_tol
        self.border_dist_tol = border_dist_tol
        self.rotation_tol = rotation_tol
        self.translation_tol = translation_tol
        self.neighbor_classifier = neighbor_classifier

    @cached_property
    def score(self) -> float:
        """Score of this cluster."""
        return self.scorer(self)

    @classmethod
    def get_default_config(cls) -> dict:
        """Default configuration for a cluster."""
        cluster_config = {
            "border_dist_tol": 5,
            "self_intersection_tol": 0.01,
            "rotation_tol": 0.17,
            "translation_tol": 30,
            "neighbor_classifier": BorderLengthNeighborClassifier(15, 5),
        }
        return cluster_config

    @property
    def piece_ids(self) -> set[str]:
        """Set of IDs of all pieces in the cluster."""
        return set(self.pieces.keys())

    def transform(self, transformation: Transformation) -> Cluster:
        """Apply a transformation to this cluster."""
        new_pieces = {
            key: piece.transform(transformation) for key, piece in self.pieces.items()
        }
        new_cluster = Cluster(
            new_pieces,
            self.scorer,
            self.self_intersection_tol,
            self.border_dist_tol,
            self.rotation_tol,
            self.translation_tol,
            self.neighbor_classifier,
            parents=[self],
        )
        return new_cluster

    @cached_property
    def self_intersection(self) -> float:
        """ """
        polygons = self.transformed_polygons
        return max(  # type: ignore
            [
                p1.intersection(p2).area / min(p1.area, p2.area)
                for p1, p2 in combinations(polygons, 2)
            ]
        )

    @property
    def transformed_polygons(self) -> list[Polygon]:
        return [piece.polygon for piece in self.pieces.values()]

    @cached_property
    def polygon_union(self) -> Polygon:
        polygons = self.transformed_polygons
        polygons = [polygon.buffer(1) for polygon in polygons]
        return unary_union(polygons)

    def _fix_overlapping_pieces(self, pieces_to_keep: set[str]) -> Cluster:
        new_pieces = self.pieces.copy()
        self_intersection_tol = self.self_intersection_tol * (
            np.log2(len(new_pieces)) + 1
        )

        for key1, key2 in combinations(self.pieces.keys(), 2):
            p1 = self.pieces[key1].polygon
            p2 = self.pieces[key2].polygon
            if p1.intersection(p2).area / min(p1.area, p2.area) > self_intersection_tol:
                if key1 not in pieces_to_keep:
                    new_pieces.pop(key1, None)
                if key2 not in pieces_to_keep:
                    new_pieces.pop(key2, None)

        if len(new_pieces) == 0:
            raise SelfIntersectionError(
                f"Self intersection {self.self_intersection} "
                f"is higher than tolerance {self_intersection_tol}"
            )
        return Cluster(
            new_pieces,
            self.scorer,
            self.self_intersection_tol,
            self.border_dist_tol,
            self.rotation_tol,
            self.translation_tol,
            self.neighbor_classifier,
            parents=None,
        )

    def find_unifying_transform(
        self, other: Cluster
    ) -> tuple[Transformation | None, Transformation | None]:
        """Find transformations which unify the common pieces of two clusters.

        Parameters
        ----------
        other

        Returns
        -------
        t1, t2
            Transformations such that when `t1` is applied to `self` and `t2` to the
            `other`, the common matches perfectly overlap (if such transformations
            exist)
        """
        common_keys = list(self.piece_ids.intersection(other.piece_ids))

        if len(common_keys) == 0:
            return None, None

        common_keys.sort(key=lambda key: self.pieces[key].polygon.area, reverse=True)
        common_key = common_keys.pop()

        return (
            self.pieces[common_key].transformation.inverse(),
            other.pieces[common_key].transformation.inverse(),
        )

    def merge(
        self,
        other: Cluster,
        finetune_iters: int = 3,
        try_fix: bool = True,
    ) -> Cluster:
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
        was_fixed = False
        if len(other.pieces) == 2:
            try_fix = False

        t1, t2 = self.find_unifying_transform(other)
        if t1 is None or t2 is None:
            raise DisjunctClustersError(
                f"Pieces {self.piece_ids} and {other.piece_ids} "
                "have no common elements."
            )

        cluster1 = self.transform(t1)
        cluster2 = other.transform(t2)

        parents = [cluster1, cluster2]

        common_keys = list(self.piece_ids.intersection(other.piece_ids))
        for key in common_keys:
            if not cluster1.pieces[key].transformation.is_close(
                cluster2.pieces[key].transformation,
                self.rotation_tol,
                self.translation_tol,
            ):
                if not try_fix:
                    raise ConflictingTransformationsError(
                        f"Transformations {cluster1.pieces[key].transformation} and "
                        f"{cluster2.pieces[key].transformation} are not close."
                    )
                parents = [cluster1]
                was_fixed = True
                cluster2.pieces.pop(key)

        if not was_fixed and (
            other.piece_ids.issubset(self.piece_ids)
            or self.piece_ids.issubset(other.piece_ids)
        ):
            raise MergeError("Clusters are subsets of each other.")

        new_pieces = cluster1.pieces
        new_pieces.update(cluster2.pieces)
        new_cluster = Cluster(
            new_pieces,
            self.scorer,
            self.self_intersection_tol,
            self.border_dist_tol,
            self.rotation_tol,
            self.translation_tol,
            self.neighbor_classifier,
            parents,
        )

        if finetune_iters > 0:
            new_cluster = new_cluster.finetune_transformations(finetune_iters)

        self_intersection_tol = self.self_intersection_tol * (
            np.log2(len(new_cluster.pieces)) + 1
        )
        if new_cluster.self_intersection > self_intersection_tol:
            if not try_fix or len(cluster2.pieces) == 2:
                raise SelfIntersectionError(
                    f"Self intersection {new_cluster.self_intersection} "
                    f"is higher than tolerance {self_intersection_tol}"
                )
            was_fixed = True

            new_cluster = new_cluster._fix_overlapping_pieces(self.piece_ids)
            new_cluster.parents = [cluster1]

        if was_fixed:
            # New cluster may be disconnected
            components = connected_components(new_cluster.graph)
            if len(components) != 1:
                major_component = components[np.argmax([len(c) for c in components])]
                piece_ids = [list(new_cluster.piece_ids)[i] for i in major_component]
                new_pieces = {key: new_cluster.pieces[key] for key in piece_ids}

                new_cluster = Cluster(
                    new_pieces,
                    self.scorer,
                    self.self_intersection_tol,
                    self.border_dist_tol,
                    self.rotation_tol,
                    self.translation_tol,
                    self.neighbor_classifier,
                    parents=None,
                )

        return new_cluster

    def common_pieces_match(self, other: Cluster) -> bool:
        """Checks whether two clusters can be merged.

        Returns True if they share at least one piece and the relative position of
        the shared pieces is the same (within some tolerance).
        """
        common_keys = self.piece_ids.intersection(other.piece_ids)
        if len(common_keys) == 0:
            return False

        t1, t2 = self.find_unifying_transform(other)
        if t1 is None or t2 is None:
            return False

        cluster1 = self.transform(t1)
        cluster2 = other.transform(t2)

        return all(
            cluster1.pieces[key].transformation.is_close(
                cluster2.pieces[key].transformation,
                self.rotation_tol,
                self.translation_tol,
            )
            for key in common_keys
        )

    def can_be_merged(self, other: Cluster) -> bool:
        """Checks whether two clusters can be merged.

        Returns True if they share at least one piece and the relative position of
        the shared pieces is the same (within some tolerance) and if they won't
        overlap after merging.

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

        t1, t2 = self.find_unifying_transform(other)
        if t1 is None or t2 is None:
            return False

        cluster1 = self.transform(t1)
        cluster2 = other.transform(t2)

        if not all(
            cluster1.pieces[key].transformation.is_close(
                cluster2.pieces[key].transformation,
                self.rotation_tol,
                self.translation_tol,
            )
            for key in common_keys
        ):
            return False

        new_pieces = cluster1.pieces
        new_pieces.update(cluster2.pieces)
        new_cluster = Cluster(
            new_pieces,
            self.scorer,
            self.self_intersection_tol,
            self.border_dist_tol,
            self.rotation_tol,
            self.translation_tol,
            self.neighbor_classifier,
            [self, other],
        )

        if new_cluster.self_intersection > self.self_intersection_tol:
            return False
        return new_cluster.self_intersection < self.self_intersection_tol

    def finetune_transformations(self, num_iters: int = 3) -> Cluster:
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
        contour_dict = {key: piece.contour for key, piece in self.pieces.items()}

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

        return Cluster(
            new_pieces,
            self.scorer,
            self.self_intersection_tol,
            self.border_dist_tol,
            self.rotation_tol,
            self.translation_tol,
            self.neighbor_classifier,
            self.parents,
        )

    @cached_property
    def matches_border_idxs(
        self,
    ) -> dict:
        matches_border_dict = {}

        if self.parents is not None:
            for parent in self.parents:
                parent_dict = {
                    keys: value
                    for keys, value in parent.matches_border_idxs
                    if set(keys).issubset(self.piece_ids)
                }
                matches_border_dict.update(parent_dict)

        for key1, key2 in combinations(self.piece_ids, 2):
            key1, key2 = min(key1, key2), max(key1, key2)
            if (key1, key2) in matches_border_dict.keys():
                continue
            piece1 = self.pieces[key1]
            piece2 = self.pieces[key2]

            idxs1, idxs2 = get_common_contour_idxs(
                piece1.contour,
                piece2.contour,
                self.border_dist_tol,
            )

            if len(idxs1) == 0:
                continue
            matches_border_dict[(key1, key2)] = (idxs1, idxs2)

        return matches_border_dict

    def get_match_border_idxs(
        self, key1: str, key2: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if (key1, key2) in self.matches_border_idxs.keys():
            idxs1, idxs2 = self.matches_border_idxs[(key1, key2)]
        elif (key2, key1) in self.matches_border_idxs.keys():
            idxs2, idxs1 = self.matches_border_idxs[(key2, key1)]
        else:
            return None, None

        return idxs1, idxs2

    def get_match_border_coordinates(
        self, key1: str, key2: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        idxs1, idxs2 = self.get_match_border_idxs(key1, key2)
        if idxs1 is None:
            return None, None

        coords1 = self.pieces[key1].contour[idxs1]
        coords2 = self.pieces[key2].contour[idxs2]

        return coords1, coords2

    @cached_property
    def neighbor_matrix(self) -> np.ndarray:
        piece_ids = list(self.piece_ids)
        matrix = np.full([len(self.pieces)] * 2, False)
        for i1, i2 in combinations(range(len(piece_ids)), 2):
            if self.neighbor_classifier(
                self.pieces[piece_ids[i1]], self.pieces[piece_ids[i2]]
            ):
                matrix[i1, i2] = True
                matrix[i2, i1] = True

        return matrix

    def draw(
        self,
        draw_contours: bool = False,
        draw_borders: bool = False,
        thickness: int = 1,
    ) -> np.ndarray:
        min_row, min_col, max_row, max_col = np.inf, np.inf, -np.inf, -np.inf

        piece_imgs = []
        center_positions = []
        for piece in self.pieces.values():
            transformation = piece.transformation
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

            center_orig = (
                piece.original_contour.max(axis=0) + piece.original_contour.min(axis=0)
            ) / 2
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

        def draw_contour_points(
            contours: Points,
            img: NpImage,
            color: tuple[int, int, int],
            thickness: int = 1,
        ) -> NpImage:
            contours = (np.concatenate(contours) - offset).round().astype(int)
            contours = contours[(contours[:, 0] < size[0]) & (contours[:, 1] < size[1])]
            img_contour = np.ones((size[0], size[1]))
            img_contour = draw_contour(contours, img_contour)
            if thickness > 1:
                img_contour = erosion(img_contour, disk(thickness // 2))
            img = np.where(img_contour[:, :, np.newaxis] == 0, np.array([[color]]), img)
            return img

        if draw_contours:
            contours = [piece.contour for piece in self.pieces.values()]
            img = draw_contour_points(contours, img, (0, 0, 0), thickness)

        if draw_borders:
            contours = []
            for id1, id2 in self.get_neighbor_pairs():
                contours1, contours2 = self.get_match_border_coordinates(id1, id2)
                contours.append(contours1)
                contours.append(contours2)
            img = draw_contour_points(
                np.stack(contours, axis=0), img, (1, 0, 0), thickness
            )

        return img

    @cached_property
    def graph(self) -> PyGraph:
        graph = PyGraph()
        graph.add_nodes_from(list(self.piece_ids))
        edges = np.where(self.neighbor_matrix)
        graph.add_edges_from(list(zip(edges[0], edges[1], [None] * len(edges[0]))))
        return graph

    def get_neighbor_pairs(self) -> set[frozenset[str]]:
        neighbor_pairs = set()
        for i, key1 in enumerate(self.piece_ids):
            for j, key2 in enumerate(self.piece_ids):
                if self.neighbor_matrix[i, j]:
                    neighbor_pairs.add(frozenset({key1, key2}))
        return neighbor_pairs

    def to_dict(self) -> dict:
        """Return this cluster as a dictionary."""
        dict_repr = {"transformed_pieces": [], "neighbors": []}
        for piece_id, piece in self.pieces.items():
            piece_dict = {
                "id": piece_id,
                "transformation": piece.transformation.to_dict(),
            }
            dict_repr["transformed_pieces"].append(piece_dict)

        for neighbor_pair in self.get_neighbor_pairs():
            dict_repr["neighbors"].append(list(neighbor_pair))
        return dict_repr

    @classmethod
    def from_dict(
        cls,
        config: dict,
        pieces: dict,
        scorer: ClusterScorerBase,
        self_intersection_tol: float,
        border_dist_tol: float,
        rotation_tol: float,
        translation_tol: float,
        neighbor_classifier: NeighborClassifierBase,
    ) -> Cluster:
        transformed_pieces = {
            p["id"]: TransformedPiece(
                pieces[p["id"]], Transformation.from_dict(p["transformation"])
            )
            for p in config["transformed_pieces"]
        }
        return cls(
            transformed_pieces,
            scorer,
            self_intersection_tol,
            border_dist_tol,
            rotation_tol,
            translation_tol,
            neighbor_classifier,
        )
