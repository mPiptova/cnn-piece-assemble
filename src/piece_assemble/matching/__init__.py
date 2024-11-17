from __future__ import annotations

from itertools import combinations
from multiprocessing import Pool
from typing import TYPE_CHECKING

import numpy as np
from more_itertools import flatten
from tqdm import tqdm

from piece_assemble.matching.match import Match

if TYPE_CHECKING:
    from piece_assemble.descriptor import DescriptorExtractor
    from piece_assemble.piece import Piece


def find_matches(
    piece1: Piece,
    piece2: Piece,
    descriptor_extractor: DescriptorExtractor,
) -> list[Match]:
    desc_dist = descriptor_extractor.dist(piece1, piece2)

    flat_dist = desc_dist.flatten()
    ordering = np.argsort(flat_dist)
    idxs1, idxs2 = np.unravel_index(ordering, desc_dist.shape)
    return [
        Match(
            dist=desc_dist[i, j],
            piece1=piece1,
            piece2=piece2,
            idxs1=np.array(piece1.segments[i].interval),
            idxs2=np.array(piece2.segments[j].interval)[::-1],
        )
        for i, j in zip(idxs1, idxs2)
        if not np.isinf(desc_dist[i, j])
    ]


def find_all_matches(
    pieces: list[Piece], descriptor_extractor: DescriptorExtractor
) -> list[Match]:
    matches = []
    for desc1, desc2 in combinations(pieces, 2):
        matches.extend(find_matches(desc1, desc2, descriptor_extractor)[:50])
    matches.sort(key=lambda match: match.dist)
    return matches


def _filter_initial(pair_matches):
    return [match for match in pair_matches if match.is_initial_transform_valid()]


def find_all_matches_with_preprocessing(
    pieces: list[Piece],
    descriptor_extractor: DescriptorExtractor,
    n_processes: int = 4,
) -> list[Match]:
    matches = [
        find_matches(piece1, piece2, descriptor_extractor)[:50]
        for piece1, piece2 in combinations(pieces, 2)
    ]
    with Pool(n_processes) as p:
        matches = p.map(
            _filter_initial,
            tqdm(matches, "Filtering matches based on initial transformation"),
        )

    matches = list(flatten(matches))
    matches.sort(key=lambda match: match.dist)
    return matches
