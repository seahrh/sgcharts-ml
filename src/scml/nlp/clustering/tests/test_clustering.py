import math
from typing import List

import pytest

from scml.nlp.clustering import *


@pytest.fixture
def docs() -> List[str]:
    return ["10 11 12 13", "13 12 11 10", "10 20 21 22", "10 11 12 23", "99"]


class TestKeywordMining:
    def test_1_gram(self, docs):
        expected = [
            Keyword(score=1.0, text="99"),
            Keyword(score=0.672, text="23"),
            Keyword(score=0.59072196, text="13"),
            Keyword(score=0.5490363, text="22"),
            Keyword(score=0.5490363, text="21"),
            Keyword(score=0.5490363, text="20"),
            Keyword(score=0.49035257, text="12"),
            Keyword(score=0.49035257, text="11"),
            Keyword(score=0.41250002, text="10"),
        ]
        it = keywords(docs=docs, ngram_range=(1, 1))
        for i in range(len(expected)):
            actual = next(it)
            assert math.isclose(expected[i].score, actual.score, rel_tol=1e-5)
            assert expected[i].text == actual.text

    def test_2_gram(self, docs):
        expected = [
            Keyword(score=0.659118, text="12 13"),
            Keyword(score=0.659118, text="12 23"),
            Keyword(score=0.57735026, text="20 21"),
            Keyword(score=0.57735026, text="10 20"),
            Keyword(score=0.57735026, text="13 12"),
            Keyword(score=0.57735026, text="12 11"),
            Keyword(score=0.57735026, text="21 22"),
            Keyword(score=0.57735026, text="11 10"),
            Keyword(score=0.53177226, text="10 11"),
            Keyword(score=0.53177226, text="11 12"),
        ]
        it = keywords(docs=docs, ngram_range=(2, 2))
        for i in range(len(expected)):
            actual = next(it)
            assert math.isclose(expected[i].score, actual.score, rel_tol=1e-5)
            assert expected[i].text == actual.text


class TestTfIdfClustering:
    def test_no_prune_edges(self, docs):
        clu = TfIdfClustering(
            docs=docs,
            similarity_min=0,
        )
        assert list(clu.G.edges(data=True)) == [
            (0, 0, {"weight": 1.0}),
            (0, 1, {"weight": 1.0}),
            (0, 2, {"weight": 0.1275634765625}),
            (0, 3, {"weight": 0.59765625}),
            (1, 1, {"weight": 1.0}),
            (1, 2, {"weight": 0.1275634765625}),
            (1, 3, {"weight": 0.59765625}),
            (2, 2, {"weight": 1.0}),
            (2, 3, {"weight": 0.11712646484375}),
            (3, 3, {"weight": 1.0}),
            (4, 4, {"weight": 1.0}),
        ]
        # If resolution is less than 1, modularity favors larger communities.
        # Greater than 1 favors smaller communities.
        assert clu.greedy_modularity_communities(resolution=1) == [0, 1, 2, 3, 4]
        assert clu.greedy_modularity_communities(resolution=0.5) == [0, 0, 0, 0, 1]

    def test_prune_edges(self, docs):
        clu = TfIdfClustering(
            docs=docs,
            similarity_min=0.5,
        )
        assert list(clu.G.edges(data=True)) == [
            (0, 0, {"weight": 1.0}),
            (0, 1, {"weight": 1.0}),
            (0, 3, {"weight": 0.59765625}),
            (1, 1, {"weight": 1.0}),
            (1, 3, {"weight": 0.59765625}),
            (2, 2, {"weight": 1.0}),
            (3, 3, {"weight": 1.0}),
            (4, 4, {"weight": 1.0}),
        ]
        # If resolution is less than 1, modularity favors larger communities.
        # Greater than 1 favors smaller communities.
        assert clu.greedy_modularity_communities(resolution=1) == [0, 1, 2, 3, 4]
        assert clu.greedy_modularity_communities(resolution=0.5) == [0, 0, 1, 0, 2]
