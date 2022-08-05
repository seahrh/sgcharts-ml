from scml.nlp.clustering import *


class TestTfIdfClustering:
    def test_case_1(self):
        clu = TfIdfClustering(docs=["10 11 12", "12 11 10", "10 12 13", "99"])
        assert list(clu.G.edges(data=True)) == [
            (0, 0, {"weight": 1}),
            (0, 1, {"weight": 1}),
            (1, 1, {"weight": 1}),
            (2, 2, {"weight": 1}),
            (3, 3, {"weight": 1}),
        ]
        assert clu.greedy_modularity_communities(resolution=1) == [0, 0, 1, 2]

    def test_case_2(self):
        clu = TfIdfClustering(
            docs=["10 11 12 13 14", "14 13 12 11 10", "10 11 12 13 15", "99"]
        )
        # assert clu.sim.tolist() == []
        assert list(clu.G.edges(data=True)) == [
            (0, 0, {"weight": 1}),
            (0, 1, {"weight": 1}),
            (1, 1, {"weight": 1}),
            (2, 2, {"weight": 1}),
            (3, 3, {"weight": 1}),
        ]
        assert clu.greedy_modularity_communities(resolution=1) == [0, 0, 1, 2]
