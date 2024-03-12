from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from piece_assemble.types import Point


@dataclass
class Match:
    key1: str
    key2: str
    score: float
    rotation: float
    translation: Point

    def __eq__(self, other: Match) -> bool:
        if self.key1 != other.key1 or self.key2 != other.key2:
            return False
        return (
            abs(np.arccos(self.rotation[0, 0]) - np.arccos(other.rotation[0, 0])) < 0.02
            and np.linalg.norm(self.translation - other.translation) < 5
        )


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
        if match == filtered_matches[-1]:
            continue
        filtered_matches.append(match)

    return filtered_matches
