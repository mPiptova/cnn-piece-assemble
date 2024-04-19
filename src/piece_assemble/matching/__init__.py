from __future__ import annotations

from itertools import combinations

import numpy as np

from piece_assemble.cluster import Cluster
from piece_assemble.descriptor import DescriptorExtractor
from piece_assemble.matching.match import Match
from piece_assemble.piece import Piece


def filter_matches(matches: list[Match]):
    """Delete duplicate matches.

    This function expects the matches are already sorted by the score in the descending
    order

    Parameters
    ----------
    matches
        List of matches.

    Returns
    -------
    List of filtered matches.
    """
    filtered_matches = []
    for match in matches:
        if len(filtered_matches) == 0:
            filtered_matches.append(match)
            continue
        if match.is_similar(filtered_matches[-1]):
            continue
        filtered_matches.append(match)

    return filtered_matches


def find_matches(
    piece1: Piece, piece2: Piece, descriptor_extractor: DescriptorExtractor
) -> list[Match]:
    desc_dist = descriptor_extractor.dist(piece1.descriptor, piece2.descriptor)
    len1 = piece1.get_segment_lengths()
    len2 = piece2.get_segment_lengths()
    max_len = np.maximum(len1[:, np.newaxis], len2[np.newaxis, :])
    desc_dist = desc_dist / (10 * max_len)
    flat_dist = desc_dist.flatten()
    ordering = np.argsort(flat_dist)
    idxs1, idxs2 = np.unravel_index(ordering[:30], desc_dist.shape)
    return [
        Match(
            piece1.name,
            piece2.name,
            dist=desc_dist[i, j],
            piece1=piece1,
            piece2=piece2,
            index1=i,
            index2=j,
        )
        for i, j in zip(idxs1, idxs2)
    ]


def find_all_matches(
    pieces: list[Piece], descriptor_extractor: DescriptorExtractor
) -> list[Match]:
    matches = []
    for desc1, desc2 in combinations(pieces, 2):
        matches.extend(find_matches(desc1, desc2, descriptor_extractor))
    matches.sort(key=lambda match: match.dist)
    return matches


def matches_to_clusters(matches: list[Match]) -> list[Cluster]:
    clusters = [match.to_cluster() for match in list(matches["match"])]
    clusters.sort(key=lambda cluster: cluster.score, reverse=True)
    return clusters
