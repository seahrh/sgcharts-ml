from typing import Set

__all__ = ["jaccard_sim", "jaccard", "dice_sim", "dice"]


def jaccard_sim(a: Set, b: Set) -> float:
    """Jaccard similarity as defined in https://en.wikipedia.org/wiki/Jaccard_index"""
    if len(a) == 0 and len(b) == 0:
        return 1
    un = a | b
    if len(un) == 0:
        return 0
    return len(a & b) / len(un)


def jaccard(a: Set, b: Set) -> float:
    """Jaccard distance as defined in https://en.wikipedia.org/wiki/Jaccard_index"""
    return 1 - jaccard_sim(a, b)


def dice_sim(a: Set, b: Set) -> float:
    """Dice coefficient as defined in https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient"""
    if len(a) == 0 and len(b) == 0:
        return 1
    return 2 * len(a & b) / (len(a) + len(b))


def dice(a: Set, b: Set) -> float:
    """Dice distance as defined in https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Not a proper distance metric as it does not satisfy the triangle inequality.
    """
    return 1 - dice_sim(a, b)
