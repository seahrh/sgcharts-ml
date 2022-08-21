import math
from typing import List

import pytest

import scml
from scml.nlp.clustering import *

scml.seed_everything()


@pytest.fixture
def docs() -> List[str]:
    return [
        "10 11 12 13",
        "13 12 11 10",
        "12 11 10 13",  # this doc and above are duplicates
        "10 20 21 22",
        "10 11 20 21",
        "10 11 12 20",
        "20 21 22 23",
        "99",
    ]


class TestKeywordMining:
    def test_1_gram(self, docs):
        expected = [
            Keyword(score=1.0, text="99"),
            Keyword(score=0.61692667, text="23"),
            Keyword(score=0.6116949, text="22"),
            Keyword(score=0.592502, text="13"),
            Keyword(score=0.592502, text="21"),
            Keyword(score=0.5419587, text="20"),
            Keyword(score=0.5419587, text="12"),
            Keyword(score=0.47972697, text="11"),
            Keyword(score=0.42711073, text="10"),
        ]
        it = keywords(docs=docs, ngram_range=(1, 1))
        for i in range(len(expected)):
            actual = next(it)
            assert math.isclose(expected[i].score, actual.score, rel_tol=1e-5)
            assert expected[i].text == actual.text

    def test_2_gram(self, docs):
        expected = [
            Keyword(score=0.69911015, text="11 20"),
            Keyword(score=0.67034394, text="22 23"),
            Keyword(score=0.67034394, text="12 20"),
            Keyword(score=0.67034394, text="12 13"),
            Keyword(score=0.67034394, text="10 20"),
            Keyword(score=0.64485943, text="13 12"),
            Keyword(score=0.64485943, text="10 13"),
            Keyword(score=0.56180054, text="21 22"),
            Keyword(score=0.56180054, text="11 12"),
            Keyword(score=0.5404425, text="12 11"),
            Keyword(score=0.5404425, text="11 10"),
            Keyword(score=0.5055913, text="20 21"),
            Keyword(score=0.5055913, text="10 11"),
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
            (0, 2, {"weight": 1.0}),
            (0, 3, {"weight": 0.1492919921875}),
            (0, 4, {"weight": 0.379150390625}),
            (0, 5, {"weight": 0.67724609375}),
            (1, 1, {"weight": 1.0}),
            (1, 2, {"weight": 1.0}),
            (1, 3, {"weight": 0.1492919921875}),
            (1, 4, {"weight": 0.379150390625}),
            (1, 5, {"weight": 0.67724609375}),
            (2, 2, {"weight": 1.0}),
            (2, 3, {"weight": 0.1492919921875}),
            (2, 4, {"weight": 0.379150390625}),
            (2, 5, {"weight": 0.67724609375}),
            (3, 3, {"weight": 1.0}),
            (3, 4, {"weight": 0.70263671875}),
            (3, 5, {"weight": 0.406494140625}),
            (3, 6, {"weight": 0.73291015625}),
            (4, 4, {"weight": 1.0}),
            (4, 5, {"weight": 0.67724609375}),
            (4, 6, {"weight": 0.467529296875}),
            (5, 5, {"weight": 1.0}),
            (5, 6, {"weight": 0.2120361328125}),
            (6, 6, {"weight": 1.0}),
            (7, 7, {"weight": 1.0}),
        ]
        assert clu.asyn_lpa_communities() == [0, 1, 2, 3, 4, 5, 6, 7]
        # If resolution is less than 1, modularity favors larger communities.
        # Greater than 1 favors smaller communities.
        assert clu.greedy_modularity_communities(resolution=1) == [
            0,
            0,
            0,
            1,
            2,
            3,
            1,
            4,
        ]
        assert clu.greedy_modularity_communities(resolution=0.5) == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
        ]

    def test_prune_edges(self, docs):
        clu = TfIdfClustering(
            docs=docs,
            similarity_min=0.5,
        )
        assert list(clu.G.edges(data=True)) == [
            (0, 0, {"weight": 1.0}),
            (0, 1, {"weight": 1.0}),
            (0, 2, {"weight": 1.0}),
            (0, 5, {"weight": 0.67724609375}),
            (1, 1, {"weight": 1.0}),
            (1, 2, {"weight": 1.0}),
            (1, 5, {"weight": 0.67724609375}),
            (2, 2, {"weight": 1.0}),
            (2, 5, {"weight": 0.67724609375}),
            (3, 3, {"weight": 1.0}),
            (3, 4, {"weight": 0.70263671875}),
            (3, 6, {"weight": 0.73291015625}),
            (4, 4, {"weight": 1.0}),
            (4, 5, {"weight": 0.67724609375}),
            (5, 5, {"weight": 1.0}),
            (6, 6, {"weight": 1.0}),
            (7, 7, {"weight": 1.0}),
        ]
        assert clu.asyn_lpa_communities() == [0, 1, 2, 3, 4, 5, 6, 7]
        # If resolution is less than 1, modularity favors larger communities.
        # Greater than 1 favors smaller communities.
        assert clu.greedy_modularity_communities(resolution=1) == [
            0,
            0,
            0,
            2,
            1,
            1,
            2,
            3,
        ]
        assert clu.greedy_modularity_communities(resolution=0.5) == [
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            2,
        ]
