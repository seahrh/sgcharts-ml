from typing import Set

__all__ = ["jaccard_sim", "jaccard"]


def jaccard_sim(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity as defined in https://en.wikipedia.org/wiki/Jaccard_index"""
    if len(a) == 0 and len(b) == 0:
        return 1
    un = a | b
    if len(un) == 0:
        return 0
    return len(a & b) / len(un)


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard distance as defined in https://en.wikipedia.org/wiki/Jaccard_index"""
    return 1 - jaccard_sim(a, b)
