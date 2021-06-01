from scml.nlp import (
    split,
    count_digit,
    count_space,
    count_alpha,
    count_upper,
    count_punctuation,
    word_ngrams,
    sentences,
)


class TestCountDigit:
    def test_case_1(self):
        assert count_digit(" aA!") == 0
        assert count_digit(" a12A!") == 2


class TestCountSpace:
    def test_case_1(self):
        assert count_space("a1A!") == 0
        assert count_space(" a1A! ") == 2


class TestCountAlpha:
    def test_case_1(self):
        assert count_alpha(" !") == 0
        assert count_alpha(" a1A!") == 2


class TestCountUpper:
    def test_case_1(self):
        assert count_upper(" a1!") == 0
        assert count_upper(" Ba1A!") == 2


class TestCountPunctuation:
    def test_case_1(self):
        assert count_punctuation(" a1A") == 0
        assert count_punctuation(" ?a1A!") == 2


class TestSplit:
    def test_delimiter_length_equals_1(self):
        assert split(delimiters=["a"], s="a1a2a",) == ["", "1", "2", ""]
        assert split(delimiters=["a", "b"], s="ab1ba2ab",) == [
            "",
            "",
            "1",
            "",
            "2",
            "",
            "",
        ]

    def test_delimiter_length_greater_than_1(self):
        assert split(
            delimiters=["a", "...", "(c)"],
            s="stackoverflow (c) is awesome... isn't it?",
        ) == ["st", "ckoverflow ", " is ", "wesome", " isn't it?"]

    def test_punctuation(self):
        assert split(
            delimiters=["!", ".", "?", ")", "("],
            s="hi there! greetings. how are you? (foo) end",
        ) == ["hi there", " greetings", " how are you", " ", "foo", " end"]


class TestWordNgrams:
    def test_separator(self):
        assert word_ngrams("hello,world", n=1, sep=",") == [("hello",), ("world",)]

    def test_gram_number(self):
        assert word_ngrams("hello world foo bar", n=1) == [
            ("hello",),
            ("world",),
            ("foo",),
            ("bar",),
        ]
        assert word_ngrams("hello world foo bar", n=2) == [
            ("hello", "world"),
            ("world", "foo"),
            ("foo", "bar"),
        ]
        assert word_ngrams("hello world foo bar", n=3) == [
            ("hello", "world", "foo"),
            ("world", "foo", "bar"),
        ]
        assert word_ngrams("hello world foo bar", n=4) == [
            ("hello", "world", "foo", "bar"),
        ]
        assert word_ngrams("hello world foo bar", n=5) == []

    def test_skip_set(self):
        assert word_ngrams("hello world foo bar", n=2, skip={"hello", "foo"}) == [
            ("world", "bar"),
        ]


class TestSentences:
    def test_end_of_sentence_punctuation(self):
        assert sentences("Full stop. Question mark? Exclamation mark! The end.") == [
            "Full stop.",
            "Question mark?",
            "Exclamation mark!",
            "The end.",
        ]

    def test_salutations(self):
        assert sentences("Mr. Huckleberry Finn met Dr. Watson for coffee.") == [
            "Mr. Huckleberry Finn met Dr. Watson for coffee."
        ]

    def test_period_delimited_strings(self):
        assert sentences("foo 123.456 bar") == ["foo 123.456 bar"]
        assert sentences("foo 123.456.789.123 bar") == ["foo 123.456.789.123 bar"]
        assert sentences("foo abc.def.ghk bar") == ["foo abc.def.ghk bar"]
