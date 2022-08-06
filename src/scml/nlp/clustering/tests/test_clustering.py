from typing import List

import pytest

from scml.nlp.clustering import *


@pytest.fixture
def docs() -> List[str]:
    return ["10 11 12 13", "13 12 11 10", "10 20 21 22", "10 11 12 23", "99"]


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
