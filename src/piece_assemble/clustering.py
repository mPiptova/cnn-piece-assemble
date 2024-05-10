import math
import os
import random
import shutil
import time
from ast import Match
from itertools import combinations
from multiprocessing import Manager, Pool, Queue

import numpy as np
from tqdm import tqdm

from piece_assemble.cluster import Cluster, ClusterScorer
from piece_assemble.descriptor import DescriptorExtractor
from piece_assemble.image import np_to_pil
from piece_assemble.matching import find_all_matches_with_preprocessing
from piece_assemble.piece import Piece


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

    def find_candidate_matches(self, n_matches: int = 40000) -> None:
        self.all_matches = find_all_matches_with_preprocessing(
            self.pieces, self.descriptor_extractor
        )[:n_matches]

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
        if self.all_matches is None:
            self.find_candidate_matches(n_used_matches)

        matches = self.all_matches.copy()
        matches.sort(key=lambda m: np.random.normal(m.dist, 0.5))

        manager = Manager()
        queue = manager.Queue()

        with Pool(n_processes) as p:
            batch_size = 100
            self._worker_count = int(np.ceil(len(matches) / batch_size))
            for j in range(0, len(matches), batch_size):
                p.apply_async(
                    matches_checker,
                    args=(
                        matches[j : min(j + batch_size, len(matches))],
                        queue,
                        self.cluster_scorer,
                        cluster_config,
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
        return self.best_cluster

    def run_iteration(
        self,
        i: int,
        trusted_cluster_config: dict,
        n_new_matches: int,
        min_complexity: float,
        queue: Queue,
    ):
        print(f"ITERATION {i}")

        max_cluster_size = 2 ** (i + 2)
        new_pair_clusters = self.get_new_pair_clusters(
            n_new_matches, queue, trusted_cluster_config, min_complexity
        )

        potential_trusted_clusters = []
        for all_found_pair_for_keys in self.all_pair_clusters.values():
            for pair_cluster in all_found_pair_for_keys:
                if pair_cluster["count"] >= 6:
                    potential_trusted_clusters.append(pair_cluster["cluster"])

        self.update_trusted_clusters(potential_trusted_clusters, lambda _: True)

        if self._store_trusted_clusters:
            self.store_iteration(f"{i}trusted", self.trusted_clusters)

        new_pair_clusters.sort(key=lambda cluster: cluster.score, reverse=True)

        if self._store_new_matches:
            self.store_iteration(f"{i}new_matches", new_pair_clusters)

        new_pair_clusters = self.apply_trusted_clusters(new_pair_clusters)

        # Get interesting previous clusters
        previous_clusters = []
        if (
            len(self.clusters) >= 1
            and len(self.clusters[0].piece_ids) > 0.5 * len(self.all_ids)
            or len(new_pair_clusters) < 5
        ):
            print("USING PREVIOUSLY ADDED CLUSTERS")
            previous_clusters = self.find_applicable_previous_clusters(
                self.clusters[0], max(5, n_new_matches - len(new_pair_clusters))
            )
            if len(previous_clusters) != 0 and self._store_old_matches:
                self.store_iteration(f"{i}old_matches", previous_clusters)
        self.clusters = self.recombine(
            self.clusters + new_pair_clusters + previous_clusters, max_cluster_size
        )

        self.clusters.sort(key=lambda cluster: cluster.score, reverse=True)

        self.clusters = self.cluster_selection(self.clusters)

        self.store_iteration(f"{i}iter", self.clusters)
        self.cluster_history.append(self.clusters)

        if self._worker_count == self._workers_finished and len(previous_clusters) == 0:
            self.assembled = True
        if len(self.best_cluster.piece_ids) == len(self.all_ids):
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
            while (
                len(new_pair_clusters) < n_new_matches
                and self._workers_finished < self._worker_count
            ):
                if queue.empty():
                    time.sleep(1)
                    continue

                new_cluster = queue.get()

                if new_cluster is None:
                    self._workers_finished += 1
                    continue
                new_cluster = self.process_new_cluster(
                    new_cluster, trusted_cluster_config, min_complexity
                )
                if new_cluster is not None:
                    new_pair_clusters.append(new_cluster)
                    pbar.update(1)
            return new_pair_clusters

    def process_new_cluster(self, new_cluster, trusted_cluster_config, min_complexity):
        if new_cluster.complexity < 1:
            return

        new_cluster_list = self.update_trusted_clusters(
            [new_cluster],
            lambda cluster: cluster_can_be_trusted(cluster, **trusted_cluster_config),
        )

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

        if new_cluster.complexity >= min_complexity:
            # Use this match now only if it's interesting enough, otherwise
            # just remember it to use it later
            return new_cluster

    def combine(
        self, cluster1: Cluster, cluster2: Cluster, max_cluster_size: int | None
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
        if self.random.random() < 0.5:
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

        piece_union = cluster1.piece_ids.union(cluster2.piece_ids)
        innovative_merge = True

        if not innovative_merge:
            return None

        finetune_iters = 0
        if len(piece_union) < 5:
            finetune_iters = 5
        elif len(piece_union) < 10:
            finetune_iters = 3

        finetune_iters = 10
        try:
            new_cluster = cluster1.merge(cluster2, finetune_iters=finetune_iters)
        except Exception:
            return None
        return new_cluster

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
            cluster_combinations = list(combinations(clusters, 2))
            random.shuffle(cluster_combinations)

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
                            used_pieces.update({key1, key2})
                            new_clusters_dict[new_key] = new_cluster
                            new_clusters_added = True

            if len(new_clusters_dict.values()) == len(clusters):
                new_clusters_added = False

            prev_cluster_len = len(clusters)
            clusters = list(new_clusters_dict.values())
            clusters = self.cluster_selection(clusters)
            if prev_cluster_len == len(clusters):
                new_clusters_added = False

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
        clusters.sort(key=lambda cluster: cluster.score, reverse=True)
        clusters_by_piece = {key: [] for key in self.all_ids}

        for i, cluster in enumerate(clusters):
            for piece_id in cluster.piece_ids:
                clusters_by_piece[piece_id].append(i)

        selected_cluster_idxs = set()

        for cluster_list in clusters_by_piece.values():
            if len(cluster_list) == 0:
                continue
            new_i = cluster_list.pop(0)

            for i in selected_cluster_idxs.copy():
                if clusters[new_i].piece_ids.issubset(clusters[i].piece_ids):
                    new_i = None
                    break
                if clusters[i].piece_ids.issubset(clusters[new_i].piece_ids):
                    selected_cluster_idxs.remove(i)
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
        new_clusters = []
        for cluster in tqdm(clusters, "Applying trusted clusters"):
            for trusted_cluster in self.trusted_clusters:
                if trusted_cluster.piece_ids.issubset(
                    cluster.piece_ids
                ) and cluster.can_be_merged(trusted_cluster):
                    continue
                if cluster.piece_ids.isdisjoint(trusted_cluster.piece_ids):
                    continue
                try:
                    cluster = trusted_cluster.merge(cluster, try_fix=True)
                except Exception:
                    cluster = None
                    break

            if cluster is None:
                continue
            for new_cluster in new_clusters.copy():
                if cluster.piece_ids.issubset(new_cluster.piece_ids):
                    cluster = None
                    break
                if new_cluster.piece_ids.issubset(cluster.piece_ids):
                    new_clusters.remove(new_cluster)
            if cluster is not None:
                new_clusters.append(cluster)

        return new_clusters

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

    def find_applicable_previous_clusters(
        self, best_cluster: Cluster, max_count: int
    ) -> list[Cluster]:
        previous_cluster_list = []
        for keys, pair_previous_clusters in self.all_pair_clusters.items():
            if len(keys.intersection(best_cluster.piece_ids)) == 1:
                previous_cluster_list.extend(
                    [cluster["cluster"] for cluster in pair_previous_clusters]
                )
        if len(previous_cluster_list) > 0:
            previous_clusters = list(
                np.random.choice(
                    previous_cluster_list, min(max_count, len(previous_cluster_list))
                )
            )
            previous_clusters = self.apply_trusted_clusters(previous_clusters)
            return previous_clusters
        return []

    def store_iteration(self, name: str, clusters: list[Cluster]) -> None:
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
            img = np_to_pil(cluster.draw())
            img.save(get_image_path(i, cluster))


def cluster_can_be_trusted(
    cluster: Cluster, complexity_threshold, dist_threshold, color_threshold
):
    return (
        cluster.complexity > complexity_threshold * (len(cluster.piece_ids) - 1)
        and cluster.dist < dist_threshold
        and cluster.color_dist < color_threshold
    )


def matches_checker(
    matches: list[Match],
    queue: Queue,
    cluster_scorer: ClusterScorer,
    config: dict,
    icp_max_iters: int,
    icp_min_change: 0.5,
) -> None:
    try:
        for match in matches:
            match = match.verify(
                config["border_dist_tol"],
                icp_max_iters=icp_max_iters,
                icp_min_change=icp_min_change,
            )
            if match is not None and match.valid:
                cluster = match.to_cluster(cluster_scorer, **config)
                cluster = cluster.finetune_transformations(3)
                if cluster.complexity >= 1:
                    queue.put(cluster)
    except Exception as e:
        print(e.with_traceback())
    finally:
        queue.put(None)
