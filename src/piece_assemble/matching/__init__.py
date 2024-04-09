from __future__ import annotations

from piece_assemble.matching.match import Match


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
