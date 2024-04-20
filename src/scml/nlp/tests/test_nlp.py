import string

from scml.nlp import (
    SYMBOL_STRING,
    MatchResult,
    RepeatingCharacter,
    RepeatingSubstring,
    collapse_whitespace,
    count_alpha,
    count_digit,
    count_punctuation,
    count_space,
    count_upper,
    find_email,
    find_url,
    has_1a1d,
    ngrams,
    replace_email,
    replace_url,
    sentences,
    split,
    strip_ip_address,
    strip_punctuation,
    strip_spans,
    strip_symbol,
    strip_xml,
    to_str,
)


class TestToStr:
    def test_case_1(self):
        assert to_str("a1") == "a1"

    # noinspection PyTypeChecker
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
        assert collapse_whitespace("a") == "a"

    def test_convert_whitespace_to_space_char(self):
        assert collapse_whitespace("1\t2") == "1 2"
        assert collapse_whitespace("1\r2") == "1 2"
        assert collapse_whitespace("1\n2") == "1 2"
        assert collapse_whitespace("1\f2") == "1 2"
        assert collapse_whitespace("1\t2\r3\n4\f5") == "1 2 3 4 5"
        assert collapse_whitespace("1\t 2\r 3\n 4\f 5") == "1 2 3 4 5"

    def test_string_is_trimmed_on_both_ends(self):
        assert collapse_whitespace(" ") == ""
        assert collapse_whitespace("\t\r\n\f") == ""
        assert collapse_whitespace("\na \t\r\n\fb\n") == "a b"
        assert collapse_whitespace(" a \t\r\n\fb ") == "a b"


class TestRepeatingCharacter:
    def test_count(self):
        assert RepeatingCharacter.count("") == 0
        assert RepeatingCharacter.count("a") == 0
        assert RepeatingCharacter.count("aa") == 1
        assert RepeatingCharacter.count("1") == 0
        assert RepeatingCharacter.count("11") == 0
        for p in string.punctuation:
            assert RepeatingCharacter.count(p) == 0
            assert RepeatingCharacter.count(p * 2) == 1
        assert RepeatingCharacter.count("aa\n\naa\t\t!!") == 3

    def test_no_replacement(self):
        max_times = 2
        f = RepeatingCharacter(max_times=max_times, letters=True, punctuation=True)
        assert f.collapse("") == ""
        assert f.collapse("a") == "a"
        assert f.collapse("aa") == "aa"
        for p in string.punctuation:
            inp = p * max_times
            assert f.collapse(inp) == inp

    def test_repeating_letter(self):
        f = RepeatingCharacter(max_times=2, letters=True, punctuation=False)
        assert f.collapse("aaa") == "aa"
        assert f.collapse("aaabbb") == "aabb"
        assert f.collapse("abbba") == "abba"
        assert f.collapse("abbba abbba") == "abba abba"

    def test_repeating_letter_is_case_preserving(self):
        f = RepeatingCharacter(max_times=2, letters=True, punctuation=False)
        assert f.collapse("AAA") == "AA"

    def test_repeating_punctuation(self):
        max_times = 2
        f = RepeatingCharacter(max_times=max_times, letters=False, punctuation=True)
        for p in string.punctuation:
            inp = p * (max_times + 1)
            e = p * max_times
            assert f.collapse(inp) == e
        assert f.collapse("a!!! b??? ***c*** --->d") == "a!! b?? **c** -->d"


class TestRepeatingSubstring:
    def test_count(self):
        f = RepeatingSubstring(
            min_length=2,
            max_times=1,
            letters=True,
            punctuation=True,
            whitespace=True,
        )
        assert f.count("") == 0
        assert f.count("\n") == 0
        assert f.count("\n\n") == 0
        assert f.count("\n\n\n") == 0
        assert f.count("a") == 0
        assert f.count("aa") == 0
        assert f.count("aaa") == 0
        assert f.count("ab ab") == 0
        assert f.count("abab") == 1
        assert f.count("ababab") == 2
        assert f.count("abababab") == 3
        assert f.count("ab cdab cd") == 1
        assert f.count("ab cdab cdab cd") == 2
        assert f.count(" ab cd ab cd") == 1
        assert f.count(" ab cd ab cd ab cd") == 2
        assert f.count("ab?cd!ab?cd!") == 1
        assert f.count("ab?cd!ab?cd!ab?cd!") == 2
        assert f.count("ab? cd!ab? cd!") == 1
        assert f.count("ab? cd!ab? cd!ab? cd!") == 2
        assert f.count(" ab? cd! ab? cd! ab? cd!") == 2

    def test_count_char(self):
        f = RepeatingSubstring(
            min_length=2,
            max_times=1,
            letters=True,
            punctuation=True,
            whitespace=True,
        )
        assert f.count_char("") == 0
        assert f.count_char("\n") == 0
        assert f.count_char("\n\n") == 0
        assert f.count_char("\n\n\n") == 0
        assert f.count_char("a") == 0
        assert f.count_char("aa") == 0
        assert f.count_char("aaa") == 0
        assert f.count_char("ab ab") == 0
        assert f.count_char("abab") == 2
        assert f.count_char("ababab") == 4
        assert f.count_char("abababab") == 6
        assert f.count_char("ab cdab cd") == 5
        assert f.count_char("ab cdab cdab cd") == 10
        assert f.count_char(" ab cd ab cd") == 6
        assert f.count_char(" ab cd ab cd ab cd") == 12
        assert f.count_char("ab?cd!ab?cd!") == 6
        assert f.count_char("ab?cd!ab?cd!ab?cd!") == 12
        assert f.count_char("ab? cd!ab? cd!") == 7
        assert f.count_char("ab? cd!ab? cd!ab? cd!") == 14
        assert f.count_char(" ab? cd! ab? cd! ab? cd!") == 16

    def test_no_replacement(self):
        min_length = 2
        max_times = 1
        f = RepeatingSubstring(
            min_length=min_length,
            max_times=max_times,
            letters=True,
            punctuation=True,
            whitespace=True,
        )
        assert f.collapse("") == ""
        assert f.collapse("\n") == "\n"
        assert f.collapse("\n\n") == "\n\n"
        assert f.collapse("\n\n\n") == "\n\n\n"
        assert f.collapse("a") == "a"
        assert f.collapse("aa") == "aa"
        assert f.collapse("aaa") == "aaa"
        assert f.collapse("ab ab") == "ab ab"
        for p in string.punctuation:
            inp = (p * min_length) * max_times
            assert f.collapse(inp) == inp

    def test_repeating_letter(self):
        f = RepeatingSubstring(
            min_length=2,
            max_times=1,
            letters=True,
            punctuation=False,
            whitespace=True,
        )
        assert f.collapse("abab") == "ab"
        assert f.collapse("ab cdab cd") == "ab cd"
        assert f.collapse(" ab cd ab cd") == " ab cd"

    def test_repeating_letter_is_case_preserving(self):
        f = RepeatingSubstring(
            min_length=2,
            max_times=1,
            letters=True,
            punctuation=False,
            whitespace=False,
        )
        assert f.collapse("ABAB") == "AB"

    def test_repeating_punctuation(self):
        min_length = 2
        max_times = 1
        f = RepeatingSubstring(
            min_length=min_length,
            max_times=max_times,
            letters=False,
            punctuation=True,
            whitespace=True,
        )
        for p in string.punctuation:
            e = p * min_length
            inp = e * (max_times + 1)
            assert f.collapse(inp) == e
        assert f.collapse("!?!?") == "!?"
        assert f.collapse("!? $#!? $#") == "!? $#"
        assert f.collapse(" !? $# !? $#") == " !? $#"

    def test_all_allowed_chars(self):
        f = RepeatingSubstring(
            min_length=2,
            max_times=1,
            letters=True,
            punctuation=True,
            whitespace=True,
        )
        assert f.collapse("ab?cd!ab?cd!") == "ab?cd!"
        assert f.collapse("ab? cd!ab? cd!") == "ab? cd!"
        assert f.collapse(" ab? cd! ab? cd!") == " ab? cd!"


class TestSplit:
    def test_delimiter_length_equals_1(self):
        assert split(
            delimiters=["a"],
            s="a1a2a",
        ) == ["", "1", "2", ""]
        assert split(
            delimiters=["a", "b"],
            s="ab1ba2ab",
        ) == [
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
            delimiters=["!", ".", "?", ")", "(", ","],
            s="hi, there! greetings. how are you? (foo) end",
        ) == ["hi", " there", " greetings", " how are you", " ", "foo", " end"]


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


class TestStripPunctuation:
    def test_no_replacement(self):
        assert strip_punctuation("") == ""
        assert strip_punctuation("a1") == "a1"

    def test_replacement(self):
        for p in string.punctuation:
            assert strip_punctuation(p) == ""


class TestStripSymbol:
    def test_no_replacement(self):
        assert strip_symbol("") == ""
        assert strip_symbol("a1") == "a1"

    def test_replacement(self):
        for c in SYMBOL_STRING:
            assert strip_symbol(c) == ""


class TestStripSpans:
    def test_strip_spans_without_replacement(self):
        assert (
            strip_spans(
                s="0123456789",
                positions=[
                    (4, 6),
                ],
                replacements=None,
            )
            == "01236789"
        )
        assert (
            strip_spans(
                s="0123456789",
                positions=[(0, 1), (4, 6), (9, 10)],
                replacements=None,
            )
            == "123678"
        )

    def test_strip_spans_with_replacement(self):
        assert (
            strip_spans(
                s="0123456789",
                positions=[
                    (4, 6),
                ],
                replacements=["__A__"],
            )
            == "0123__A__6789"
        )
        assert (
            strip_spans(
                s="0123456789",
                positions=[(0, 1), (4, 6), (9, 10)],
                replacements=["__A__", "__B__", "__C__"],
            )
            == "__A__123__B__678__C__"
        )

    def test_ignore_invalid_spans(self):
        assert (
            strip_spans(
                s="0123456789",
                positions=[(-1, -1), (-1, 1), (1, -1), (4, 4), (10, 10)],
                replacements=None,
            )
            == "0123456789"
        )


class TestStripXml:
    def test_no_replacement(self):
        assert strip_xml("") == ""
        assert strip_xml("a") == "a"
        assert strip_xml("1 < 2 and 2 > 1") == "1 < 2 and 2 > 1"
        assert strip_xml("1<2 and 2>1") == "1<2 and 2>1"

    def test_replacement(self):
        assert strip_xml("<strong>a</strong>") == "a"
        assert strip_xml("<p>a</p><p>b</p>") == "ab"
        assert strip_xml("<br />") == ""


class TestFindUrl:

    def test_empty_string(self):
        assert find_url("") == []

    def test_no_matches(self):
        assert find_url("foo1 bar2 co") == []
        assert find_url("http://foo1,bar2.co") == []

    def test_whole_string_match(self):
        assert find_url("http://foo1.bar2") == [
            MatchResult(match="http://foo1.bar2", start=0, end=16)
        ]

    def test_single_match_start(self):
        assert find_url("http://foo1.bar2 bar") == [
            MatchResult(match="http://foo1.bar2", start=0, end=16)
        ]

    def test_single_match_mid(self):
        assert find_url("foo http://foo1.bar2 bar") == [
            MatchResult(match="http://foo1.bar2", start=4, end=20)
        ]

    def test_single_match_end(self):
        assert find_url("foo http://foo1.bar2") == [
            MatchResult(match="http://foo1.bar2", start=4, end=20)
        ]

    def test_multi_match_start(self):
        assert find_url("http://foo1.bar2 foo http://bar3.foo4 bar") == [
            MatchResult(match="http://foo1.bar2", start=0, end=16),
            MatchResult(match="http://bar3.foo4", start=21, end=37),
        ]

    def test_multi_match_mid(self):
        assert find_url("foo http://foo1.bar2 foo http://bar3.foo4 bar") == [
            MatchResult(match="http://foo1.bar2", start=4, end=20),
            MatchResult(match="http://bar3.foo4", start=25, end=41),
        ]

    def test_multi_match_end(self):
        assert find_url("foo http://foo1.bar2 foo http://bar3.foo4") == [
            MatchResult(match="http://foo1.bar2", start=4, end=20),
            MatchResult(match="http://bar3.foo4", start=25, end=41),
        ]

    def test_non_overlapping_match(self):
        # TODO pattern can't split the conjoined urls properly
        assert find_url("http://foo1.bar2.cohttp://foo3.bar4.co") == [
            MatchResult(match="http://foo1.bar2.cohttp://foo3.bar4.co", start=0, end=38)
        ]


class TestReplaceUrl:
    def test_no_replacement(self):
        assert replace_url("") == ""
        assert replace_url("a") == "a"
        assert replace_url(".com") == ".com"
        assert replace_url("a.com") == "a.com"
        assert replace_url("www.a") == "www.a"
        assert replace_url("sub1.a.com") == "sub1.a.com"
        assert replace_url("www.a#.com") == "www.a#.com"
        assert replace_url("www.a-.com") == "www.a-.com"
        assert replace_url("www.-a.com") == "www.-a.com"
        assert replace_url("http://www.a") == "http://www.a"
        assert replace_url("http://a") == "http://a"
        assert replace_url("s3://a.com") == "s3://a.com"
        assert replace_url("a.com/dir1") == "a.com/dir1"
        assert replace_url("a.com/file.html") == "a.com/file.html"

    def test_scheme_and_domain_name(self):
        assert replace_url("http://a.com") == ""
        assert replace_url("https://a.com") == ""
        assert replace_url("https://mp3.com") == ""
        assert replace_url("1 https://mp3.com 2") == "1  2"

    def test_subdomain(self):
        assert replace_url("www.a.com") == ""
        assert replace_url("www.mp3.com") == ""
        assert replace_url("1 www.mp3.com 2") == "1  2"
        assert replace_url("http://www.a.com") == ""
        assert replace_url("https://www.a.com") == ""
        assert replace_url("https://www.mp3.com") == ""
        assert replace_url("1 https://www.mp3.com 2") == "1  2"
        assert replace_url("http://sub1.a.com") == ""
        assert replace_url("https://sub1.a.com") == ""
        assert replace_url("https://sub1.mp3.com") == ""
        assert replace_url("1 https://sub1.mp3.com 2") == "1  2"
        assert replace_url("http://sub2.sub1.a.com") == ""
        assert replace_url("https://sub2.sub1.a.com") == ""
        assert replace_url("https://sub2.sub1.mp3.com") == ""
        assert replace_url("1 https://sub2.sub1.mp3.com 2") == "1  2"
        assert replace_url("http://sub3.sub2.sub1.a.com") == ""
        assert replace_url("https://sub3.sub2.sub1.a.com") == ""
        assert replace_url("https://sub3.sub2.sub1.mp3.com") == ""
        assert replace_url("1 https://sub3.sub2.sub1.mp3.com 2") == "1  2"

    def test_subdirectories(self):
        assert replace_url("http://a.com/dir1") == ""
        assert replace_url("https://a.com/dir1") == ""
        assert replace_url("https://mp3.com/dir1") == ""
        assert replace_url("1 https://mp3.com/dir1 2") == "1  2"
        assert replace_url("http://a.com/dir1/dir2") == ""
        assert replace_url("https://a.com/dir1/dir2") == ""
        assert replace_url("https://mp3.com/dir1/dir2") == ""
        assert replace_url("1 https://mp3.com/dir1/dir2 2") == "1  2"
        assert replace_url("http://a.com/dir1/dir2/dir3") == ""
        assert replace_url("https://a.com/dir1/dir2/dir3") == ""
        assert replace_url("https://mp3.com/dir1/dir2/dir3") == ""
        assert replace_url("1 https://mp3.com/dir1/dir2/dir3 2") == "1  2"

    def test_file_extension(self):
        assert replace_url("http://a.com/file.html") == ""
        assert replace_url("http://a.com/file.xml") == ""
        assert replace_url("http://a.com/file.pdf") == ""
        assert replace_url("http://a.com/file.json") == ""
        assert replace_url("1 http://a.com/file.html 2") == "1  2"
        assert replace_url("http://a.com/dir1/file.html") == ""
        assert replace_url("http://a.com/dir1/file.xml") == ""
        assert replace_url("http://a.com/dir1/file.pdf") == ""
        assert replace_url("http://a.com/dir1/file.json") == ""
        assert replace_url("1 http://a.com/dir1/file.html 2") == "1  2"


class TestStripIpAddress:
    def test_no_replacement(self):
        assert strip_ip_address("") == ""
        assert strip_ip_address("1.2") == "1.2"
        assert strip_ip_address("1.2.3.") == "1.2.3."
        assert strip_ip_address("256.1.2.3") == "256.1.2.3"
        assert strip_ip_address("g:h:i:j:k:l:m:n") == "g:h:i:j:k:l:m:n"

    def test_ipv4_address(self):
        assert strip_ip_address("255.1.2.3") == ""
        assert strip_ip_address("1.2.3.4") == ""
        assert strip_ip_address(".1.2.3.4.") == ".."
        assert strip_ip_address("a 1.2.3.4 b") == "a  b"

    def test_ipv6_address_case_1(self):
        assert strip_ip_address("1:2:3:4:5:6:7:8") == ""
        assert strip_ip_address("g1:2:3:4:5:6:7:8g") == "gg"

    def test_ipv6_address_case_2(self):
        assert strip_ip_address("1::") == ""
        assert strip_ip_address("1:2:3:4:5:6:7::") == ""
        assert strip_ip_address("g1:2:3:4:5:6:7::g") == "gg"

    def test_ipv6_address_case_3(self):
        assert strip_ip_address("1::8") == ""
        assert strip_ip_address("1:2:3:4:5:6::8") == ""
        assert strip_ip_address("g1:2:3:4:5:6::8g") == "gg"

    def test_ipv6_address_case_4(self):
        assert strip_ip_address("1::7:8") == ""
        assert strip_ip_address("1:2:3:4:5::7:8") == ""
        assert strip_ip_address("1:2:3:4:5::8") == ""
        assert strip_ip_address("g1:2:3:4:5::8g") == "gg"

    def test_ipv6_address_case_5(self):
        assert strip_ip_address("1::6:7:8") == ""
        assert strip_ip_address("1:2:3:4::6:7:8") == ""
        assert strip_ip_address("1:2:3:4::8") == ""
        assert strip_ip_address("g1:2:3:4::8g") == "gg"

    def test_ipv6_address_case_6(self):
        assert strip_ip_address("1::5:6:7:8") == ""
        assert strip_ip_address("1:2:3::5:6:7:8") == ""
        assert strip_ip_address("1:2:3::8") == ""
        assert strip_ip_address("g1:2:3::8g") == "gg"

    def test_ipv6_address_case_7(self):
        assert strip_ip_address("1::4:5:6:7:8") == ""
        assert strip_ip_address("1:2::4:5:6:7:8") == ""
        assert strip_ip_address("1:2::8") == ""
        assert strip_ip_address("g1:2::8g") == "gg"

    def test_ipv6_address_case_8(self):
        assert strip_ip_address("1::3:4:5:6:7:8") == ""
        assert strip_ip_address("g1::3:4:5:6:7:8g") == "gg"

    def test_ipv6_address_case_9(self):
        assert strip_ip_address("::2:3:4:5:6:7:8") == ""
        assert strip_ip_address("::8") == ""
        assert strip_ip_address("::") == ""
        assert strip_ip_address("g::8g") == "gg"

    def test_ipv6_address_case_10(self):
        # link-local IPv6 addresses with zone index
        assert strip_ip_address("fe80::7:8%eth0") == ""
        assert strip_ip_address("fe80::7:8%1") == ""
        assert strip_ip_address("gfe80::7:8%1g") == "g"

    def test_ipv6_address_case_11(self):
        # IPv4-mapped IPv6 addresses and IPv4-translated addresses
        assert strip_ip_address("::255.255.255.255") == ""
        assert strip_ip_address("::ffff:255.255.255.255") == ""
        assert strip_ip_address("g ::ffff:0:255.255.255.255 g") == "g  g"

    def test_ipv6_address_case_12(self):
        # IPv4-Embedded IPv6 Address
        assert strip_ip_address("2001:db8:3:4::192.0.2.33") == ""
        assert strip_ip_address("64:ff9b::192.0.2.33") == ""
        assert strip_ip_address("g 64:ff9b::192.0.2.33 g") == "g  g"


class TestFindEmail:

    def test_empty_string(self):
        assert find_email("") == []

    def test_no_matches(self):
        assert find_email("foo1 bar2 co") == []

    def test_whole_string_match(self):
        assert find_email("foo1@bar2.co") == [
            MatchResult(match="foo1@bar2.co", start=0, end=12)
        ]

    def test_single_match_start(self):
        assert find_email("foo1@bar2.co bar") == [
            MatchResult(match="foo1@bar2.co", start=0, end=12)
        ]

    def test_single_match_mid(self):
        assert find_email("foo foo1@bar2.co bar") == [
            MatchResult(match="foo1@bar2.co", start=4, end=16)
        ]

    def test_single_match_end(self):
        assert find_email("foo foo1@bar2.co") == [
            MatchResult(match="foo1@bar2.co", start=4, end=16)
        ]

    def test_multi_match_start(self):
        assert find_email("foo1@bar2.co foo bar3@foo4.co bar") == [
            MatchResult(match="foo1@bar2.co", start=0, end=12),
            MatchResult(match="bar3@foo4.co", start=17, end=29),
        ]

    def test_multi_match_mid(self):
        assert find_email("bar foo1@bar2.co foo bar3@foo4.co bar") == [
            MatchResult(match="foo1@bar2.co", start=4, end=16),
            MatchResult(match="bar3@foo4.co", start=21, end=33),
        ]

    def test_multi_match_end(self):
        assert find_email("bar foo1@bar2.co foo bar3@foo4.co") == [
            MatchResult(match="foo1@bar2.co", start=4, end=16),
            MatchResult(match="bar3@foo4.co", start=21, end=33),
        ]

    def test_non_overlapping_match(self):
        assert find_email("foo1@bar2.foo3@bar4.co") == [
            MatchResult(match="foo1@bar2.foo3", start=0, end=14),
        ]


class TestReplaceEmail:
    def test_empty_string(self):
        assert replace_email(s="", replacement="") == ""

    def test_no_matches(self):
        assert replace_email(s="foo1 bar2 co", replacement="") == "foo1 bar2 co"

    def test_whole_string_replaced(self):
        assert replace_email(s="foo1@bar2.co", replacement="") == ""

    def test_replacement(self):
        assert (
            replace_email(s="foo foo1@bar2.co bar", replacement="xxx") == "foo xxx bar"
        )
