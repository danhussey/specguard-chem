from __future__ import annotations

"""Simple Levenshtein edit-distance implementation."""

from typing import Sequence


def levenshtein(a: Sequence[str] | str, b: Sequence[str] | str) -> int:
    """Return the Levenshtein distance between *a* and *b*.

    This implementation is iterative and uses a single rolling row, so it can
    handle reasonably sized SMILES strings without significant overhead.
    """

    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]
