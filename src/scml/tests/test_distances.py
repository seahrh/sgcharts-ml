import math

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
        a = {-1}
        b = {1, 2, 3, 4}
        assert jaccard_sim(a, b) == 0
        assert jaccard(a, b) == 1
        assert jaccard_sim(b, a) == 0
        assert jaccard(b, a) == 1
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


class TestDice:
    def test_both_sets_empty(self):
        a = set()
        b = set()
        assert dice_sim(a, b) == 1
        assert dice(a, b) == 0

    def test_one_empty_set_and_one_nonempty_set(self):
        a = set()
        b = {1}
        assert dice_sim(a, b) == 0
        assert dice(a, b) == 1
        assert dice_sim(b, a) == 0
        assert dice(b, a) == 1

    def test_both_sets_nonempty(self):
        a = {-1}
        b = {1, 2, 3, 4}
        assert dice_sim(a, b) == 0
        assert dice(a, b) == 1
        assert dice_sim(b, a) == 0
        assert dice(b, a) == 1
        a = {1}
        b = {1, 2, 3, 4}
        assert dice_sim(a, b) == 0.4
        assert dice(a, b) == 0.6
        assert dice_sim(b, a) == 0.4
        assert dice(b, a) == 0.6
        a = {1, 2}
        b = {1, 2, 3, 4}
        assert math.isclose(dice_sim(a, b), 2 / 3)
        assert math.isclose(dice(a, b), 1 / 3)
        assert math.isclose(dice_sim(b, a), 2 / 3)
        assert math.isclose(dice(b, a), 1 / 3)
        a = {1, 2, 3}
        b = {1, 2, 3, 4}
        assert math.isclose(dice_sim(a, b), 6 / 7)
        assert math.isclose(dice(a, b), 1 / 7)
        assert math.isclose(dice_sim(b, a), 6 / 7)
        assert math.isclose(dice(b, a), 1 / 7)
        a = {1, 2, 3, 4}
        b = {1, 2, 3, 4}
        assert dice_sim(a, b) == 1
        assert dice(a, b) == 0
        assert dice_sim(b, a) == 1
        assert dice(b, a) == 0
