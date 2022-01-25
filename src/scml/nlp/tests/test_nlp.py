from scml.nlp import (
    to_str,
    split,
    count_digit,
    count_space,
    count_alpha,
    count_upper,
    count_punctuation,
    ngrams,
    sentences,
    has_1a1d,
    emoji_shortcode_to_text,
    collapse_whitespace,
)


class TestToStr:
    def test_case_1(self):
        assert to_str("a1") == "a1"

    def test_numerics_cast_to_string(self):
        assert to_str(1) == "1"
        assert to_str(1.2) == "1.2"


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


class TestCollapseWhitespace:
    def test_no_replacement(self):
        assert collapse_whitespace("") == ""
        assert collapse_whitespace(" ") == " "
        assert collapse_whitespace("a") == "a"

    def test_case_1(self):
        assert collapse_whitespace("\t\r\n\f") == " "
        assert collapse_whitespace("1\t\r\n\f2") == "1 2"
        assert collapse_whitespace(" \t \r \n \f ") == " "


class TestSplit:
    def test_delimiter_length_equals_1(self):
        assert (
            split(
                delimiters=["a"],
                s="a1a2a",
            )
            == ["", "1", "2", ""]
        )
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
        assert (
            split(
                delimiters=["a", "...", "(c)"],
                s="stackoverflow (c) is awesome... isn't it?",
            )
            == ["st", "ckoverflow ", " is ", "wesome", " isn't it?"]
        )

    def test_punctuation(self):
        assert (
            split(
                delimiters=["!", ".", "?", ")", "(", ","],
                s="hi, there! greetings. how are you? (foo) end",
            )
            == ["hi", " there", " greetings", " how are you", " ", "foo", " end"]
        )


class TestNgrams:
    def test_gram_number(self):
        assert ngrams(["hello", "world", "foo", "bar"], n=1) == [
            ("hello",),
            ("world",),
            ("foo",),
            ("bar",),
        ]
        assert ngrams(["hello", "world", "foo", "bar"], n=2) == [
            ("hello", "world"),
            ("world", "foo"),
            ("foo", "bar"),
        ]
        assert ngrams(["hello", "world", "foo", "bar"], n=3) == [
            ("hello", "world", "foo"),
            ("world", "foo", "bar"),
        ]
        assert ngrams(["hello", "world", "foo", "bar"], n=4) == [
            ("hello", "world", "foo", "bar"),
        ]
        assert ngrams(["hello", "world", "foo", "bar"], n=5) == []

    def test_skip_set(self):
        assert ngrams(["hello", "world", "foo", "bar"], n=2, skip={"hello", "foo"}) == [
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


class TestHasAtLeastOneDigitAndOneLetter:
    def test_no_matches(self):
        assert not has_1a1d("")
        assert not has_1a1d("A")
        assert not has_1a1d("a")
        assert not has_1a1d("1")
        assert not has_1a1d("Aa")
        assert not has_1a1d("aA")
        assert not has_1a1d("12")
        assert not has_1a1d("1.2")
        assert not has_1a1d("1,234")

    def test_matches(self):
        assert has_1a1d("A1")
        assert has_1a1d("a1")
        assert has_1a1d("1A")
        assert has_1a1d("1a")
        assert has_1a1d("10x20")

    def test_include_chars(self):
        include = ":-"
        assert has_1a1d("a-1", include=include)
        assert has_1a1d("A-1", include=include)
        assert has_1a1d("1-a", include=include)
        assert has_1a1d("1-A", include=include)
        assert has_1a1d("a:1", include=include)
        assert has_1a1d("A:1", include=include)
        assert has_1a1d("1:a", include=include)
        assert has_1a1d("1:A", include=include)
        assert has_1a1d("-a1", include=include)
        assert has_1a1d("a1-", include=include)
        assert has_1a1d(":a1", include=include)
        assert has_1a1d("a1:", include=include)
        # Allow only chars inside the whitelist
        assert not has_1a1d(",a1", include=include)
        assert not has_1a1d("a,1", include=include)
        assert not has_1a1d("a1,", include=include)
        # Missing either letter or digit
        assert not has_1a1d('15"', include='"')


class TestEmojiShortcodeToText:
    def test_no_matches(self):
        assert emoji_shortcode_to_text("") == ""
        assert emoji_shortcode_to_text(":a=1:") == ":a=1:"

    def test_single_match(self):
        assert emoji_shortcode_to_text("1 :joy: 2") == "1 (joy) 2"
        assert (
            emoji_shortcode_to_text("1 :person_in_suit_levitating_light_skin_tone: 2")
            == "1 (person in suit levitating light skin tone) 2"
        )
        assert (
            emoji_shortcode_to_text("1 :person-in-suit-levitating-light-skin-tone: 2")
            == "1 (person in suit levitating light skin tone) 2"
        )
        assert (
            emoji_shortcode_to_text("1 :person in suit levitating light skin tone: 2")
            == "1 (person in suit levitating light skin tone) 2"
        )

    def test_multiple_matches(self):
        assert emoji_shortcode_to_text("1 :joy: 2 :foo: 3") == "1 (joy) 2 (foo) 3"
        assert (
            emoji_shortcode_to_text(
                "1 :person_in_suit_levitating_light_skin_tone: 2 :foo_bar: 3"
            )
            == "1 (person in suit levitating light skin tone) 2 (foo bar) 3"
        )
        assert (
            emoji_shortcode_to_text(
                "1 :person-in-suit-levitating-light-skin-tone: 2 :foo-bar: 3"
            )
            == "1 (person in suit levitating light skin tone) 2 (foo bar) 3"
        )
        assert (
            emoji_shortcode_to_text(
                "1 :person in suit levitating light skin tone: 2 :foo bar: 3"
            )
            == "1 (person in suit levitating light skin tone) 2 (foo bar) 3"
        )
