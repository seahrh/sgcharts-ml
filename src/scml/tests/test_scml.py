from scml import *


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
