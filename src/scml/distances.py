from typing import Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

__all__ = ["jaccard_sim", "jaccard", "dice_sim", "dice", "sharpened_cosine_similarity"]


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


def sharpened_cosine_similarity(
    a: np.ndarray, b: np.ndarray, p: float = 1
) -> np.ndarray:
    """The cosine is known for being broad, that is,
    two quite different vectors can have a moderately high cosine similarity.
    It can be sharpened by raising the magnitude of the result to a power, p, while maintaining the sign.

    Based on https://github.com/brohrer/sharpened-cosine-similarity/tree/main
    """
    if p < 1:
        raise ValueError("power p must be greater than or equal 1.")
    cs: np.ndarray = cosine_similarity(a, b)
    sign = np.sign(cs)
    return sign * (np.abs(cs) ** p)
