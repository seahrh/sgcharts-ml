from scml.distances import *


class TestJaccard:
    def test_both_sets_empty(self):
        a = set()
        b = set()
        assert jaccard_sim(a, b) == 1
        assert jaccard(a, b) == 0

    def test_one_empty_set_and_one_nonempty_set(self):
        a = set()
        b = {1}
        assert jaccard_sim(a, b) == 0
        assert jaccard(a, b) == 1
        assert jaccard_sim(b, a) == 0
        assert jaccard(b, a) == 1

    def test_both_sets_nonempty(self):
        a = {1}
        b = {1, 2, 3, 4}
        assert jaccard_sim(a, b) == 0.25
        assert jaccard(a, b) == 0.75
        assert jaccard_sim(b, a) == 0.25
        assert jaccard(b, a) == 0.75
        a = {1, 2}
        b = {1, 2, 3, 4}
        assert jaccard_sim(a, b) == 0.5
        assert jaccard(a, b) == 0.5
        assert jaccard_sim(b, a) == 0.5
        assert jaccard(b, a) == 0.5
        a = {1, 2, 3}
        b = {1, 2, 3, 4}
        assert jaccard_sim(a, b) == 0.75
        assert jaccard(a, b) == 0.25
        assert jaccard_sim(b, a) == 0.75
        assert jaccard(b, a) == 0.25
        a = {1, 2, 3, 4}
        b = {1, 2, 3, 4}
        assert jaccard_sim(a, b) == 1
        assert jaccard(a, b) == 0
        assert jaccard_sim(b, a) == 1
        assert jaccard(b, a) == 0
