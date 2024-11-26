from __future__ import annotations

import math
import os
import random
import shutil
import time
from multiprocessing import Manager, Pool
from typing import TYPE_CHECKING

import numpy as np
from skimage.transform import rescale
from tqdm import tqdm

from image import np_to_pil
from piece_assemble.cluster import Cluster

if TYPE_CHECKING:
    from piece_assemble.cluster import ClusterScorer
    from piece_assemble.descriptor import DescriptorExtractor
    from piece_assemble.matching import Match
    from piece_assemble.piece import Piece


class Queue:
    def __init__(self, parallel: bool = False):
        self.parallel = parallel
        if parallel:
            self._queue = Manager().Queue()

        else:
            self._queue = []

    def empty(self):
        if self.parallel:
            return self._queue.empty()
        else:
            return len(self._queue) == 0

    def get(self):
        if self.parallel:
            return self._queue.get()
        else:
            return self._queue.pop(0)

    def put(self, item):
        if self.parallel:
            self._queue.put(item)
        else:
            self._queue.append(item)


class Clustering:
    """Class for assembling pieces into clusters."""

    def __init__(
        self,
        pieces: list[Piece],
        descriptor_extractor: DescriptorExtractor,
        cluster_scorer: ClusterScorer,
    ) -> None:
        self.pieces = pieces
        self.descriptor_extractor = descriptor_extractor
        self.cluster_scorer = cluster_scorer
        self.all_ids = [piece.name for piece in pieces]

        self.reset(True)
        self.set_logging(None)
        self.random = random.Random()

    @property
    def best_cluster(self) -> Cluster | None:
        if len(self.clusters) == 0:
            return None
        return self.clusters[0]

    def reset(self, reset_all_matches: bool = False) -> None:
        self.clusters = []
        self.trusted_clusters = []
        self._i = 0
        self.all_pair_clusters = {}
        self.cluster_history = []
        self.assembled = False

        if reset_all_matches:
            self.all_matches = None

    def set_logging(
        self,
        output_images_path: str,
        store_new_matches: bool = True,
        store_old_matches: bool = False,
        store_trusted_clusters: bool = False,
    ) -> None:
        self._output_path = output_images_path
        self._store_new_matches = store_new_matches
        self._store_old_matches = store_old_matches
        self._store_trusted_clusters = store_trusted_clusters

        try:
            os.mkdir(output_images_path)
        except Exception:
            pass

    def find_candidate_matches(self, n_matches: int = 40000) -> None:
        self.all_matches = self.descriptor_extractor.find_all_matches(self.pieces)[
            :n_matches
        ]

    def __call__(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
        n_new_matches: int = 10,
        n_processes: int = 4,
        min_complexity: int = 2,
        n_used_matches: int = 40000,
    ) -> Cluster | None:
        self.cluster_config = cluster_config

        if self.all_matches is None:
            self.find_candidate_matches(n_used_matches)

        # matches = self.all_matches.copy()

        queue = Queue(parallel=n_processes > 1)

        if n_processes > 1:
            self._run_parallel(
                n_iters,
                trusted_cluster_config,
                icp_max_iters,
                icp_min_change,
                n_new_matches,
                n_processes,
                min_complexity,
                queue,
            )
        else:
            self._run_serial(
                n_iters,
                trusted_cluster_config,
                icp_max_iters,
                icp_min_change,
                n_new_matches,
                min_complexity,
                queue,
            )
        return self.best_cluster

    def _run_parallel(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
        n_new_matches: int,
        n_processes: int,
        min_complexity: int,
        queue: Queue,
    ):
        matches = self.all_matches.copy()
        with Pool(n_processes) as p:
            self._worker_count = n_processes
            for j in range(0, self._worker_count):
                p.apply_async(
                    matches_checker,
                    args=(
                        matches[j :: self._worker_count],
                        queue,
                        self.cluster_config,
                        icp_max_iters,
                        icp_min_change,
                    ),
                )

            self._workers_finished = 0
            for i in range(self._i, self._i + n_iters):
                if self.assembled:
                    break
                self._i = i + 1
                self.run_iteration(
                    i,
                    trusted_cluster_config,
                    n_new_matches,
                    min_complexity,
                    queue,
                )

    def _run_serial(
        self,
        n_iters: int,
        trusted_cluster_config: dict,
        icp_max_iters: int,
        icp_min_change: float,
        n_new_matches: int,
        min_complexity: int,
        queue: Queue,
    ):
        for match in tqdm(self.all_matches, desc="Generating matches"):
            compact_match = match.verify(
                self.cluster_config["border_dist_tol"],
                icp_max_iters=icp_max_iters,
                icp_min_change=icp_min_change,
            )
            if compact_match is not None:
                queue.put(compact_match)

        self._worker_count = 1
        self._workers_finished = 0

        for i in range(self._i, self._i + n_iters):
            if self.assembled:
                break
            self._i = i + 1
            self.run_iteration(
                i,
                trusted_cluster_config,
                n_new_matches,
                min_complexity,
                queue,
            )

    def run_iteration(
        self,
        i: int,
        trusted_cluster_config: dict,
        n_new_matches: int,
        min_complexity: float,
        queue: Queue,
    ):
        print(f"ITERATION {i}")

        new_pair_clusters = self.get_new_pair_clusters(
            n_new_matches, queue, trusted_cluster_config, min_complexity
        )

        potential_trusted_clusters = []
        for all_found_pair_for_keys in self.all_pair_clusters.values():
            for pair_cluster in all_found_pair_for_keys:
                if pair_cluster["count"] >= 10:
                    potential_trusted_clusters.append(pair_cluster["cluster"])

        self.update_trusted_clusters(potential_trusted_clusters, lambda _: True)

        if self._store_trusted_clusters:
            self.store_iteration(f"{i}trusted", self.trusted_clusters)

        new_pair_clusters.sort(key=lambda cluster: cluster.score, reverse=True)

        if self._store_new_matches:
            self.store_iteration(f"{i}new_matches", new_pair_clusters)

        # Get interesting previous clusters
        previous_clusters = []
        use_previous_clusters = len(self.clusters) >= 1 and (
            len(self.clusters[0].piece_ids) > 0.5 * len(self.all_ids)
            or len(new_pair_clusters) < 5
        )
        if use_previous_clusters:
            print("RECYCLING OLD MATCHES")
            previous_clusters = self.find_applicable_previous_clusters(
                self.clusters[0], max(5, n_new_matches - len(new_pair_clusters))
            )
            if len(previous_clusters) != 0 and self._store_old_matches:
                self.store_iteration(f"{i}old_matches", previous_clusters)
        self.clusters = self.use_new_matches(
            self.clusters + self.trusted_clusters, new_pair_clusters + previous_clusters
        )
        self.clusters = self.recombine(self.clusters)
        self.clusters.sort(key=lambda cluster: cluster.score, reverse=True)

        self.store_iteration(f"{i}iter", self.clusters)
        self.cluster_history.append(self.clusters)

        if (
            self._worker_count == self._workers_finished
            and len(previous_clusters) == 0
            and use_previous_clusters
        ):
            self.assembled = True
        if len(self.clusters) > 0 and len(self.best_cluster.piece_ids) == len(
            self.all_ids
        ):
            self.assembled = True

    def get_new_pair_clusters(
        self,
        n_new_matches: int,
        queue: Queue,
        trusted_cluster_config: dict,
        min_complexity: float,
    ) -> list[Cluster]:
        new_pair_clusters = []

        with tqdm(desc="Generating new matches", total=n_new_matches) as pbar:
            while len(new_pair_clusters) < n_new_matches:
                if self._workers_finished == self._worker_count:
                    break

                if self._worker_count == 1 and queue.empty():
                    break

                if self._worker_count > 1:
                    if queue.empty():
                        time.sleep(1)
                        continue

                new_match = queue.get()

                if new_match is None:
                    self._workers_finished += 1
                    print("Worker finished")
                    continue

                new_cluster = new_match.to_cluster(
                    self.cluster_scorer,
                    self.cluster_config,
                    pieces_dict={piece.name: piece for piece in self.pieces},
                )
                # display(np_to_pil(new_cluster.draw()))
                new_cluster = self.process_new_cluster(
                    new_cluster, trusted_cluster_config, min_complexity
                )
                if new_cluster is not None:
                    new_pair_clusters.append(new_cluster)
                    pbar.update(1)
            return new_pair_clusters

    def process_new_cluster(self, new_cluster, trusted_cluster_config, min_complexity):
        if type(min_complexity) in (int, float):
            min_complexity = [min_complexity, min_complexity]
        if new_cluster.complexity < min_complexity[1]:
            return

        new_cluster_list = self.update_trusted_clusters(
            [new_cluster],
            lambda cluster: cluster_can_be_trusted(cluster, **trusted_cluster_config),
        )
        new_cluster_list = self._check_new_clusters(new_cluster_list)

        if len(new_cluster_list) == 0:
            return
        new_cluster = new_cluster_list[0]

        new_keys = frozenset(new_cluster.piece_ids)
        is_duplicate = False
        all_found_pair_for_keys = self.all_pair_clusters.get(new_keys, [])
        for pair_cluster_dict in all_found_pair_for_keys:
            pair_cluster = pair_cluster_dict["cluster"]
            if pair_cluster.can_be_merged(new_cluster):
                if new_cluster.score > pair_cluster.score:
                    pair_cluster_dict["cluster"] = new_cluster
                pair_cluster_dict["count"] = pair_cluster_dict["count"] + 1
                is_duplicate = True
                break

        if is_duplicate:
            return

        all_found_pair_for_keys.append({"cluster": new_cluster, "count": 1})
        self.all_pair_clusters[new_keys] = all_found_pair_for_keys

        if new_cluster.complexity >= min_complexity[0]:
            # Use this match now only if it's interesting enough, otherwise
            # just remember it to use it later
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
        for c1 in tqdm(pair_clusters, desc="Using new matches"):
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
            with tqdm(desc="Recombining", total=math.comb(len(clusters), 2)) as pbar:
                for i, c1 in enumerate(clusters[:-1]):
                    key1 = frozenset(c1.piece_ids)
                    if key1 in used_pieces:
                        pbar.update(len(clusters[i + 1 :]))
                        continue
                    for c2 in clusters[i + 1 :]:
                        pbar.update(1)
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

    def cluster_selection(self, clusters: list[Cluster]):
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

    def apply_trusted_clusters(self, clusters: Cluster) -> list[Cluster]:
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
        for cluster in tqdm(clusters, "Applying trusted clusters"):
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

    def update_trusted_clusters(self, clusters, trust_function) -> list[Cluster]:
        other_clusters = []
        for cluster in clusters:
            used_trusted_clusters = []
            if not trust_function(cluster):
                other_clusters.append(cluster)
                continue
            for trusted_cluster in self.trusted_clusters:
                if cluster.piece_ids.issubset(trusted_cluster.piece_ids):
                    cluster = None
                    break
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                try:
                    cluster = cluster.merge(trusted_cluster, try_fix=False)
                    used_trusted_clusters.append(trusted_cluster)
                except Exception:
                    cluster = None
                    break

            if cluster is not None:
                self.trusted_clusters.append(cluster)
                for trusted_cluster in used_trusted_clusters:
                    self.trusted_clusters.remove(trusted_cluster)
        return other_clusters

    def _check_new_clusters(self, clusters) -> list[Cluster]:
        selected_clusters = []
        for cluster in clusters:
            can_be_used = True
            for trusted_cluster in self.trusted_clusters:
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                if not cluster.can_be_merged(trusted_cluster):
                    can_be_used = False
                    break
            if can_be_used:
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
        for keys, pair_previous_clusters in self.all_pair_clusters.items():
            if keys not in cluster_pairs:
                pair_previous_clusters = [
                    cluster["cluster"] for cluster in pair_previous_clusters
                ]
                pair_previous_clusters.sort(
                    key=lambda cluster: cluster.score, reverse=True
                )
                if keys.issubset(cluster.piece_ids) and pair_previous_clusters[
                    0
                ].common_pieces_match(cluster):
                    continue
                previous_cluster_list.append(pair_previous_clusters[0])

        previous_cluster_list = self._check_new_clusters(previous_cluster_list)
        if len(previous_cluster_list) > 0:
            worst_score = min([cluster.score for cluster in previous_cluster_list])
            probabilities = np.array(
                [
                    (cluster.score - worst_score) ** 4
                    if cluster.piece_ids.intersection(cluster.piece_ids) == 1
                    else (cluster.score - worst_score)
                    for cluster in previous_cluster_list
                ]
            )
            probabilities = probabilities**0.5 + +0.0000001
            probabilities = probabilities / sum(probabilities)

            previous_clusters = list(
                np.random.choice(
                    previous_cluster_list,
                    min(max_count, len(previous_cluster_list)),
                    replace=False,
                    p=probabilities,
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

        def get_image_path(i, cluster):
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
):
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
    return (
        cluster.complexity > complexity_threshold * (len(cluster.piece_ids) - 1)
        and cluster.dist < dist_threshold
        and cluster.color_dist < color_threshold
    )


def matches_checker(
    matches: list[Match],
    queue: Queue,
    config: dict,
    icp_max_iters: int,
    icp_min_change: 0.5,
) -> None:
    try:
        for match in matches:
            compact_match = match.verify(
                config["border_dist_tol"],
                icp_max_iters=icp_max_iters,
                icp_min_change=icp_min_change,
            )
            if compact_match is not None:
                queue.put(compact_match)
    except Exception as e:
        print(e)
    finally:
        queue.put(None)
