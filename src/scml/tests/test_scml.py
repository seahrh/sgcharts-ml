import json

import numpy as np

from scml import *


class TestNumpyEncoder:

    def test_serialize_int_and_float_arrays(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        assert (
            json.dumps(
                {"f1": b, "f2": [2, (2, 3, 4), a], "f3": [1, 2]}, cls=NumpyEncoder
            )
            == '{"f1": [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], "f2": [2, [2, 3, 4], [[1, 2, 3], [4, 5, 6]]], "f3": [1, 2]}'
        )


class TestGetBoolean:
    def test_truthy(self):
        assert getboolean("1")
        assert getboolean("yes")
        assert getboolean("true")
        assert getboolean("TRUE")
        assert getboolean("on")

    def test_falsey(self):
        assert not getboolean("")
        assert not getboolean("0")
        assert not getboolean("1 ")  # trailing whitespace
        assert not getboolean("no")
        assert not getboolean("false")
        assert not getboolean("off")


class TestWeightedChoice:
    def test_empty_weights(self):
        assert weighted_choice(weights=[]) == 0

    def test_choice_counts(self):
        counts = [0] * 4
        for _ in range(1000):
            i = weighted_choice(weights=[1, 2, 3, 4])
            counts[i] += 1
        assert counts[3] > counts[2] > counts[1] > counts[0]


class TestVarName:
    def test_when_string_contains_uppercase_chars_then_return_lowercase(self):
        assert var_name("Abc") == "abc"

    def test_when_string_contains_punctuation_then_remove_punctuation(self):
        assert var_name("a!@#$%^&*(){}[]-=_+\"'~`|\\b") == "a_b"

    def test_when_string_contains_whitespace_then_replace_with_single_underscore(self):
        assert var_name(" \n\r\t a \n\r\t b \n\r\t ") == "a_b"

    def test_when_string_is_already_legal_variable_name_then_do_nothing(self):
        assert var_name("a_b") == "a_b"


class TestContiguousRanges:
    def test_single_item(self):
        assert contiguous_ranges([0]) == [(0, 0)]

    def test_single_range(self):
        assert contiguous_ranges([0, 1]) == [(0, 1)]
        assert contiguous_ranges([0, 2, 3, 4, 6]) == [(0, 0), (2, 4), (6, 6)]

    def test_multiple_ranges(self):
        assert contiguous_ranges([0, 1, 3, 4, 5]) == [(0, 1), (3, 5)]
        assert contiguous_ranges([0, 1, 3, 5, 6, 8, 10, 11]) == [
            (0, 1),
            (3, 3),
            (5, 6),
            (8, 8),
            (10, 11),
        ]
