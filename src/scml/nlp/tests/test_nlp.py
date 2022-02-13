import string

from scml.nlp import (
    RepeatingCharacter,
    RepeatingSubstring,
    collapse_whitespace,
    count_alpha,
    count_digit,
    count_punctuation,
    count_space,
    count_upper,
    emoji_shortcode_to_text,
    has_1a1d,
    ngrams,
    sentences,
    split,
    strip_ip_address,
    strip_punctuation,
    strip_url,
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
        assert strip_xml("") == ""
        assert strip_xml("a1") == "a1"

    def test_replacement(self):
        for p in string.punctuation:
            assert strip_punctuation(p) == ""


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


class TestStripUrl:
    def test_no_replacement(self):
        assert strip_url("") == ""
        assert strip_url("a") == "a"
        assert strip_url(".com") == ".com"
        assert strip_url("a.com") == "a.com"
        assert strip_url("www.a") == "www.a"
        assert strip_url("sub1.a.com") == "sub1.a.com"
        assert strip_url("www.a#.com") == "www.a#.com"
        assert strip_url("www.a-.com") == "www.a-.com"
        assert strip_url("www.-a.com") == "www.-a.com"
        assert strip_url("http://www.a") == "http://www.a"
        assert strip_url("http://a") == "http://a"
        assert strip_url("s3://a.com") == "s3://a.com"
        assert strip_url("a.com/dir1") == "a.com/dir1"
        assert strip_url("a.com/file.html") == "a.com/file.html"

    def test_scheme_and_domain_name(self):
        assert strip_url("http://a.com") == ""
        assert strip_url("https://a.com") == ""
        assert strip_url("https://mp3.com") == ""
        assert strip_url("1 https://mp3.com 2") == "1  2"

    def test_subdomain(self):
        assert strip_url("www.a.com") == ""
        assert strip_url("www.mp3.com") == ""
        assert strip_url("1 www.mp3.com 2") == "1  2"
        assert strip_url("http://www.a.com") == ""
        assert strip_url("https://www.a.com") == ""
        assert strip_url("https://www.mp3.com") == ""
        assert strip_url("1 https://www.mp3.com 2") == "1  2"
        assert strip_url("http://sub1.a.com") == ""
        assert strip_url("https://sub1.a.com") == ""
        assert strip_url("https://sub1.mp3.com") == ""
        assert strip_url("1 https://sub1.mp3.com 2") == "1  2"
        assert strip_url("http://sub2.sub1.a.com") == ""
        assert strip_url("https://sub2.sub1.a.com") == ""
        assert strip_url("https://sub2.sub1.mp3.com") == ""
        assert strip_url("1 https://sub2.sub1.mp3.com 2") == "1  2"
        assert strip_url("http://sub3.sub2.sub1.a.com") == ""
        assert strip_url("https://sub3.sub2.sub1.a.com") == ""
        assert strip_url("https://sub3.sub2.sub1.mp3.com") == ""
        assert strip_url("1 https://sub3.sub2.sub1.mp3.com 2") == "1  2"

    def test_subdirectories(self):
        assert strip_url("http://a.com/dir1") == ""
        assert strip_url("https://a.com/dir1") == ""
        assert strip_url("https://mp3.com/dir1") == ""
        assert strip_url("1 https://mp3.com/dir1 2") == "1  2"
        assert strip_url("http://a.com/dir1/dir2") == ""
        assert strip_url("https://a.com/dir1/dir2") == ""
        assert strip_url("https://mp3.com/dir1/dir2") == ""
        assert strip_url("1 https://mp3.com/dir1/dir2 2") == "1  2"
        assert strip_url("http://a.com/dir1/dir2/dir3") == ""
        assert strip_url("https://a.com/dir1/dir2/dir3") == ""
        assert strip_url("https://mp3.com/dir1/dir2/dir3") == ""
        assert strip_url("1 https://mp3.com/dir1/dir2/dir3 2") == "1  2"

    def test_file_extension(self):
        assert strip_url("http://a.com/file.html") == ""
        assert strip_url("http://a.com/file.xml") == ""
        assert strip_url("http://a.com/file.pdf") == ""
        assert strip_url("http://a.com/file.json") == ""
        assert strip_url("1 http://a.com/file.html 2") == "1  2"
        assert strip_url("http://a.com/dir1/file.html") == ""
        assert strip_url("http://a.com/dir1/file.xml") == ""
        assert strip_url("http://a.com/dir1/file.pdf") == ""
        assert strip_url("http://a.com/dir1/file.json") == ""
        assert strip_url("1 http://a.com/dir1/file.html 2") == "1  2"


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
