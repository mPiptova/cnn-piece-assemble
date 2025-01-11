from __future__ import annotations

import os
import random
import shutil
from typing import TYPE_CHECKING, Callable

import numpy as np
from skimage.transform import rescale
from tqdm import tqdm

from image import np_to_pil
from piece_assemble.cluster import Cluster
from piece_assemble.cycles import TransformationGraph
from piece_assemble.models import PairNetwork
from piece_assemble.models.predict import get_matches

if TYPE_CHECKING:
    from piece_assemble.cluster import ClusterScorer
    from piece_assemble.matching.match import Match
    from piece_assemble.piece import Piece


class Clustering:
    """Class for assembling pieces into clusters."""

    def __init__(
        self,
        pieces: list[Piece],
        cluster_scorer: ClusterScorer,
    ) -> None:
        self.pieces = pieces
        self.cluster_scorer = cluster_scorer
        self.all_ids = [piece.name for piece in pieces]

        self.reset(True)
        self.set_logging(None)
        self.random = random.Random()

    @property
    def best_cluster(self) -> Cluster | None:
        if len(self.clusters) == 0:
            return None
        return self.clusters[0]  # type: ignore

    def reset(self, forget_found_matches: bool = False) -> None:
        self.clusters = []
        self.trusted_clusters = []
        self._i = 0
        self.used_pair_clusters = {}
        self.cluster_history = []
        self.assembled = False

        if forget_found_matches:
            self.all_pair_clusters = []

    def set_logging(
        self,
        output_images_path: str | None,
        store_new_matches: bool = True,
        store_old_matches: bool = False,
        store_trusted_clusters: bool = False,
    ) -> None:

        self._output_path = output_images_path
        self._store_new_matches = store_new_matches
        self._store_old_matches = store_old_matches
        self._store_trusted_clusters = store_trusted_clusters

        if output_images_path is None:
            return

        try:
            os.mkdir(output_images_path)
        except Exception:
            pass

    def __call__(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
        model: PairNetwork,
    ) -> Cluster | None:
        self.cluster_config = cluster_config
        self.model = model

        self.run(
            n_iters,
            trusted_cluster_config,
            icp_max_iters,
            icp_min_change,
        )
        return self.best_cluster

    def _generate_matches(self, icp_max_iters: int, icp_min_change: float) -> None:
        if len(self.all_pair_clusters) > 0:
            return

        self.all_pair_clusters = []
        self.all_matches = []

        piece_dict = {piece.name: piece for piece in self.pieces}
        candidate_matches = get_matches(self.model, piece_dict, 0.7)

        for candidate_match in tqdm(candidate_matches, desc="Finding matches"):
            verified_match = candidate_match.verify(
                self.cluster_config["border_dist_tol"],
                icp_max_iters=icp_max_iters,
                icp_min_change=icp_min_change,
            )
            if verified_match is not None:
                self.all_matches.append(verified_match)
                cluster = verified_match.to_cluster(
                    self.cluster_scorer, self.cluster_config, piece_dict
                )
                self.all_pair_clusters.append(cluster)

        self.all_pair_clusters = sorted(
            self.all_pair_clusters, key=lambda c: c.score, reverse=True
        )

    def run(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
    ) -> None:
        self._generate_matches(icp_max_iters, icp_min_change)

        clusters_queue = self.all_pair_clusters.copy()
        if len(clusters_queue) < 10 * len(self.pieces):
            cycles = self.get_cycles(self.all_matches)
            self.update_trusted_clusters(cycles, lambda _: True)
        clusters_queue = self.update_trusted_clusters(
            clusters_queue,
            lambda cluster: cluster_can_be_trusted(cluster, **trusted_cluster_config),
        )

        for i in tqdm(range(self._i, self._i + n_iters)):
            if self.assembled:
                break
            self._i = i + 1
            self.run_iteration(
                i,
                clusters_queue,
            )

    def get_cycles(self, verified_matches: list[Match]) -> list[Cluster]:
        graph = TransformationGraph.from_matches(verified_matches)
        cycles = graph.find_consistent_cycles(3) + graph.find_consistent_cycles(4)
        clusters = []
        pieces_dict = {piece.name: piece for piece in self.pieces}
        for cycle in cycles:
            cluster = graph.cycle_to_cluster(
                cycle, pieces_dict, self.cluster_scorer, self.cluster_config
            )
            if cluster is not None:
                clusters.append(cluster)
        return clusters

    def run_iteration(
        self,
        i: int,
        matches_queue: list,
    ) -> None:
        new_cluster = self.get_new_pair_cluster(matches_queue)

        if new_cluster is None:
            return

        if self._store_trusted_clusters:
            self.store_iteration(f"{i}trusted", self.trusted_clusters)

        if self._store_new_matches:
            self.store_iteration(f"{i}new_matches", [new_cluster])

        self.clusters = self.use_new_matches(
            self.clusters + self.trusted_clusters, [new_cluster]
        )

        self.clusters = self.recombine(self.clusters)
        self.clusters.sort(key=lambda cluster: cluster.score, reverse=True)

        self.store_iteration(f"{i}iter", self.clusters)
        self.cluster_history.append(self.clusters)

        if self.best_cluster is None:
            return

        if len(self.clusters) > 0 and len(self.best_cluster.piece_ids) == len(
            self.all_ids
        ):
            self.assembled = True

    def get_new_pair_cluster(
        self,
        cluster_queue: list[Cluster],
    ) -> Cluster | None:
        while True:
            new_cluster = None
            if len(cluster_queue) > 0:
                new_cluster = cluster_queue.pop(0)

            if new_cluster is None and self.best_cluster is not None:
                previous_clusters = self.find_applicable_previous_clusters(
                    self.best_cluster, 1
                )
                if len(previous_clusters) == 0 and len(cluster_queue) == 0:
                    return None
                if len(previous_clusters) != 0:
                    new_cluster = previous_clusters[0]

            if new_cluster is not None:
                new_cluster = self.process_new_cluster(new_cluster)
            if new_cluster is not None:
                return new_cluster

    def process_new_cluster(self, new_cluster: Cluster) -> Cluster | None:
        new_cluster_list = self._check_new_clusters([new_cluster])

        if len(new_cluster_list) == 0:
            return None
        new_cluster = new_cluster_list[0]

        new_keys = frozenset(new_cluster.piece_ids)
        if new_keys in self.used_pair_clusters:
            return new_cluster

        for cluster in self.clusters:
            if new_keys.issubset(
                frozenset(cluster.piece_ids)
            ) and cluster.common_pieces_match(new_cluster):
                new_clusters = self.update_trusted_clusters(
                    [new_cluster], lambda _: True
                )
                if len(new_clusters) > 0:
                    new_cluster = new_clusters[0]
                else:
                    new_cluster = None
                    break

        if new_cluster is not None:
            self.used_pair_clusters[new_keys] = new_cluster

        return new_cluster

    def combine(
        self,
        cluster1: Cluster,
        cluster2: Cluster,
        max_cluster_size: int | None,
        randomize_order: bool = True,
        finetune_iters: int | None = None,
    ) -> Cluster | None:
        """Combine two clusters, if possible.

        Parameters
        ----------
        cluster1
            First cluster.
        cluster2
            Second cluster.
        max_cluster_size
            Cluster won't be merged if their total number of pieces exceeds this limit.

        Returns
        -------
        new_cluster
            New combined cluster.
        """
        # merging depends on the cluster order, so let's randomize it
        if randomize_order and self.random.random() < 0.5:
            cluster2, cluster1 = cluster1, cluster2

        if (
            max_cluster_size is not None
            and len(cluster1.piece_ids.union(cluster2.piece_ids)) > max_cluster_size
        ):
            return None
        if cluster1.piece_ids.isdisjoint(cluster2.piece_ids):
            return None

        if cluster1.piece_ids.issubset(
            cluster2.piece_ids
        ) or cluster2.piece_ids.issubset(cluster1.piece_ids):
            return None

        if finetune_iters is None:
            finetune_iters = 5

        try:
            new_cluster = cluster1.merge(cluster2, finetune_iters=finetune_iters)
        except Exception:
            return None
        return new_cluster

    def use_new_matches(
        self,
        clusters: list[Cluster],
        pair_clusters: list[Cluster],
        max_cluster_size: int | None = None,
    ) -> list[Cluster]:
        clusters = self.apply_trusted_clusters(clusters)
        clusters = self.cluster_selection(clusters)
        pair_clusters.sort(key=lambda cluster: cluster.score, reverse=True)
        for c1 in pair_clusters:
            clusters_dict = {
                frozenset(cluster.piece_ids): cluster for cluster in clusters + [c1]
            }

            for c2 in clusters:
                new_cluster = self.combine(
                    c1, c2, max_cluster_size, randomize_order=False, finetune_iters=3
                )
                if new_cluster is None:
                    continue
                key = frozenset(new_cluster.piece_ids)
                if key in clusters_dict.keys():
                    if clusters_dict[key].score >= new_cluster.score:
                        continue

                clusters_dict[key] = new_cluster

            clusters = list(clusters_dict.values())
            clusters = self.apply_trusted_clusters(clusters)
            clusters = self.cluster_selection(clusters)
        return clusters

    def recombine(
        self, clusters: list[Cluster], max_cluster_size: int | None = None
    ) -> list[Cluster]:
        """Create new clusters from given clusters via merging.

        Merges together all clusters which can be merged.
        Runs in iterations.

        Parameters
        ----------
        clusters
            List of clusters.
        max_cluster_size
            Cluster won't be merged if their total number of pieces exceeds this limit.

        Returns
        -------
        recombined_clusters
            List of clusters created from the original clusters with merging.
        """
        new_clusters_added = True
        while new_clusters_added:
            new_clusters_added = False
            new_clusters_dict = {
                frozenset(cluster.piece_ids): cluster for cluster in clusters
            }
            self.random.shuffle(clusters)

            used_pieces = set()
            for i, c1 in enumerate(clusters[:-1]):
                key1 = frozenset(c1.piece_ids)
                if key1 in used_pieces:
                    continue
                for c2 in clusters[i + 1 :]:
                    key2 = frozenset(c2.piece_ids)
                    if key1 in used_pieces or key2 in used_pieces:
                        continue

                    new_cluster = self.combine(c1, c2, max_cluster_size)
                    if new_cluster is None:
                        continue

                    new_key = frozenset(new_cluster.piece_ids)
                    if new_key in new_clusters_dict.keys():
                        if new_clusters_dict[new_key].score < new_cluster.score:
                            new_clusters_dict[new_key] = new_cluster
                    else:
                        used_pieces.update(
                            {key for key in (key1, key2) if len(key) > 2}
                        )
                        new_clusters_dict[new_key] = new_cluster
                        new_clusters_added = True

            if len(new_clusters_dict.values()) == len(clusters):
                new_clusters_added = False

            prev_cluster_len = len(clusters)
            clusters = list(new_clusters_dict.values())
            clusters = self.cluster_selection(clusters)
            if prev_cluster_len == len(clusters):
                new_clusters_added = False

        clusters = self.apply_trusted_clusters(clusters)
        if len(clusters) == 0:
            return self.trusted_clusters
        return clusters

    def cluster_selection(self, clusters: list[Cluster]) -> list[Cluster]:
        """Select best cluster representative.

        Parameters
        ----------
        clusters
            List of clusters

        Returns
        -------
        selected_clusters
            List of selected clusters.
        """
        clusters = clusters + self.trusted_clusters
        clusters.sort(key=lambda cluster: cluster.score, reverse=True)
        clusters_by_piece = {key: [] for key in self.all_ids}

        for i, cluster in enumerate(clusters):
            for piece_id in cluster.piece_ids:
                clusters_by_piece[piece_id].append(i)

        selected_cluster_idxs = set()

        for piece_id, cluster_list in clusters_by_piece.items():
            if len(cluster_list) == 0:
                continue
            new_i = cluster_list.pop(0)

            if new_i is not None:
                selected_cluster_idxs.add(new_i)

        selection = [clusters[i] for i in selected_cluster_idxs]
        selection.sort(key=lambda cluster: cluster.score, reverse=True)
        return selection

    def apply_trusted_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        """Extend given clusters by trusted clusters.

        Parameters
        ----------
        clusters

        Returns
        -------
        new_cluster
            Extended clusters (only valid ones).
        """
        if len(self.trusted_clusters) == 0:
            return clusters
        new_clusters = {}
        for cluster in clusters:
            for trusted_cluster in self.trusted_clusters:
                if trusted_cluster.piece_ids.issubset(
                    cluster.piece_ids
                ) and trusted_cluster.common_pieces_match(cluster):
                    continue
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                try:
                    cluster = trusted_cluster.merge(
                        cluster, try_fix=True, finetune_iters=5
                    )
                except Exception:
                    cluster = None
                    break

            if cluster is None:
                continue

            key = frozenset(cluster.piece_ids)
            if key in new_clusters.keys() and new_clusters[key].score >= cluster.score:
                continue
            new_clusters[key] = cluster

        return list(new_clusters.values())

    def update_trusted_clusters(
        self, clusters: list[Cluster], trust_function: Callable
    ) -> list[Cluster]:
        other_clusters = []
        for cluster in clusters:
            used_trusted_clusters = []
            for trusted_cluster in self.trusted_clusters:
                if cluster.piece_ids.issubset(trusted_cluster.piece_ids):
                    cluster = None
                    break
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                if trust_function(cluster):
                    try:
                        cluster = cluster.merge(trusted_cluster, try_fix=False)
                        used_trusted_clusters.append(trusted_cluster)
                    except Exception:
                        cluster = None
                        break

            if cluster is None:
                continue

            if trust_function(cluster) or len(used_trusted_clusters) > 0:
                self.trusted_clusters.append(cluster)
                for trusted_cluster in used_trusted_clusters:
                    self.trusted_clusters.remove(trusted_cluster)

            else:
                other_clusters.append(cluster)

        return other_clusters

    def _check_new_clusters(self, clusters: list[Cluster]) -> list[Cluster]:
        selected_clusters = []
        for cluster in clusters:
            can_be_used = True
            for trusted_cluster in self.trusted_clusters:
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                if not cluster.can_be_merged(trusted_cluster):
                    can_be_used = False
                    break
            if can_be_used and cluster is not None:
                selected_clusters.append(cluster)
        return selected_clusters

    def find_applicable_previous_clusters(
        self, cluster: Cluster, max_count: int
    ) -> list[Cluster]:
        """
        Finds the suitable previously find pair clusters for the given cluster.

        Parameters
        ----------
        best_cluster
            The best cluster.
        max_count
            The maximum number of clusters to return.

        Returns
        -------
        list
            A list of applicable previous clusters.
        """

        previous_cluster_list = []
        cluster_pairs = cluster.get_neighbor_pairs()
        for keys, c in self.used_pair_clusters.items():
            if keys not in cluster_pairs:
                if keys.issubset(cluster.piece_ids) and c.common_pieces_match(cluster):
                    continue
                previous_cluster_list.append(c)

        previous_cluster_list = self._check_new_clusters(previous_cluster_list)
        if len(previous_cluster_list) > 0:
            previous_clusters = list(
                np.random.choice(
                    previous_cluster_list,
                    min(max_count, len(previous_cluster_list)),
                    replace=False,
                )
            )
            return previous_clusters
        return []

    def store_iteration(self, name: str, clusters: list[Cluster]) -> None:
        """
        Store the given clusters in the specified directory.

        Parameters
        ----------
        name
            The name of the directory.
        clusters
            The list of clusters to be stored.

        """
        if self._output_path is None:
            return

        base_dir = f"{self._output_path}/{name}"

        try:
            os.mkdir(base_dir)
        except Exception:
            shutil.rmtree(base_dir)
            os.mkdir(base_dir)

        def get_image_path(i: int, cluster: Cluster) -> str:
            image_name = (
                f"{name}_{i:03d}_score{cluster.score:.2f}"
                + f"_color{cluster.color_dist:.3f}_dist{cluster.dist:.3f}"
                + f"_complexity{cluster.complexity:.3f}.png"
            )
            return os.path.join(base_dir, image_name)

        for i, cluster in enumerate(clusters):
            img = np_to_pil(rescale(cluster.draw(), 0.5, channel_axis=2))
            img.save(get_image_path(i, cluster))


def cluster_can_be_trusted(
    cluster: Cluster,
    complexity_threshold: float,
    dist_threshold: float,
    color_threshold: float,
) -> bool:
    """
    Check if a cluster can be trusted.

    Parameters
    ----------
    cluster
        The cluster to be checked.
    complexity_threshold
        The threshold for the cluster's complexity.
    dist_threshold
        The threshold for the cluster's distance.
    color_threshold
        The threshold for the cluster's color distance.

    Returns
    -------
    bool
        True if the cluster can be trusted, False otherwise.

    """
    return (  # type: ignore
        cluster.complexity > complexity_threshold * (len(cluster.piece_ids) - 1)
        and cluster.dist < dist_threshold
        and cluster.color_dist < color_threshold
    )
