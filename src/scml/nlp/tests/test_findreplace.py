import string

from scml.nlp.findreplace import *
from scml.nlp.findreplace import SYMBOL_STRING


class TestReplacePunctuation:
    def test_no_replacement(self):
        assert replace_punctuation("") == ""
        assert replace_punctuation("a1") == "a1"

    def test_replacement(self):
        for p in string.punctuation:
            assert replace_punctuation(p) == ""


class TestReplaceSymbol:
    def test_no_replacement(self):
        assert replace_symbol("") == ""
        assert replace_symbol("a1") == "a1"

    def test_replacement(self):
        for c in SYMBOL_STRING:
            assert replace_symbol(c) == ""


class TestReplaceSpans:
    def test_strip_spans_without_replacement(self):
        assert (
            replace_spans(
                s="0123456789",
                positions=[
                    (4, 6),
                ],
                replacements=None,
            )
            == "01236789"
        )
        assert (
            replace_spans(
                s="0123456789",
                positions=[(0, 1), (4, 6), (9, 10)],
                replacements=None,
            )
            == "123678"
        )

    def test_strip_spans_with_replacement(self):
        assert (
            replace_spans(
                s="0123456789",
                positions=[
                    (4, 6),
                ],
                replacements=["__A__"],
            )
            == "0123__A__6789"
        )
        assert (
            replace_spans(
                s="0123456789",
                positions=[(0, 1), (4, 6), (9, 10)],
                replacements=["__A__", "__B__", "__C__"],
            )
            == "__A__123__B__678__C__"
        )

    def test_ignore_invalid_spans(self):
        assert (
            replace_spans(
                s="0123456789",
                positions=[(-1, -1), (-1, 1), (1, -1), (4, 4), (10, 10)],
                replacements=None,
            )
            == "0123456789"
        )


class TestFindXml:

    def test_empty_string(self):
        assert find_xml("") == []

    def test_no_matches(self):
        assert find_xml("foo1 bar2 co") == []
        assert find_xml("http://foo1,bar2.co") == []

    def test_whole_string_match(self):
        assert find_xml("<br>") == [MatchResult(match="<br>", start=0, end=4)]
        assert find_xml("</p>") == [MatchResult(match="</p>", start=0, end=4)]

    def test_single_match_start(self):
        assert find_xml("<br> bar") == [MatchResult(match="<br>", start=0, end=4)]
        assert find_xml("</p> bar") == [MatchResult(match="</p>", start=0, end=4)]

    def test_single_match_mid(self):
        assert find_xml("foo <br> bar") == [MatchResult(match="<br>", start=4, end=8)]
        assert find_xml("foo </p> bar") == [MatchResult(match="</p>", start=4, end=8)]

    def test_single_match_end(self):
        assert find_xml("foo <br>") == [MatchResult(match="<br>", start=4, end=8)]
        assert find_xml("foo </p>") == [MatchResult(match="</p>", start=4, end=8)]

    def test_multi_match_start(self):
        assert find_xml("<br> foo </p> bar") == [
            MatchResult(match="<br>", start=0, end=4),
            MatchResult(match="</p>", start=9, end=13),
        ]

    def test_multi_match_mid(self):
        assert find_xml("foo <br> foo </p> bar") == [
            MatchResult(match="<br>", start=4, end=8),
            MatchResult(match="</p>", start=13, end=17),
        ]

    def test_multi_match_end(self):
        assert find_xml("foo <br> bar </p>") == [
            MatchResult(match="<br>", start=4, end=8),
            MatchResult(match="</p>", start=13, end=17),
        ]

    def test_non_overlapping_match(self):
        assert find_xml("<br></p>") == [
            MatchResult(match="<br>", start=0, end=4),
            MatchResult(match="</p>", start=4, end=8),
        ]
        assert find_xml("</p><br>") == [
            MatchResult(match="</p>", start=0, end=4),
            MatchResult(match="<br>", start=4, end=8),
        ]


class TestReplaceXml:
    def test_no_replacement(self):
        assert replace_xml("") == ""
        assert replace_xml("a") == "a"
        assert replace_xml("1 < 2 and 2 > 1") == "1 < 2 and 2 > 1"
        assert replace_xml("1<2 and 2>1") == "1<2 and 2>1"

    def test_replacement(self):
        assert replace_xml("<strong>a</strong>") == "a"
        assert replace_xml("<p>a</p><p>b</p>") == "ab"
        assert replace_xml("<br />") == ""


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
        # note conjoined urls with no delimiter cannot be separated
        # the 2 urls must be separated by whitespace
        assert find_url("http://foo1.bar2 http://bar3.foo4") == [
            MatchResult(match="http://foo1.bar2", start=0, end=16),
            MatchResult(match="http://bar3.foo4", start=17, end=33),
        ]
        assert find_url("http://bar3.foo4 http://foo1.bar2") == [
            MatchResult(match="http://bar3.foo4", start=0, end=16),
            MatchResult(match="http://foo1.bar2", start=17, end=33),
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


class TestFindIpAddress:

    def test_empty_string(self):
        assert find_ip_address("") == []

    def test_no_matches(self):
        assert find_ip_address("1234abcd") == []

    def test_whole_string_match(self):
        assert find_ip_address("1.2.3.4") == [
            MatchResult(match="1.2.3.4", start=0, end=7)
        ]
        assert find_ip_address("1:2:3:4:a:b:c:d") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=0, end=15)
        ]

    def test_single_match_start(self):
        assert find_ip_address("1.2.3.4 bar") == [
            MatchResult(match="1.2.3.4", start=0, end=7)
        ]
        assert find_ip_address("1:2:3:4:a:b:c:d bar") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=0, end=15)
        ]

    def test_single_match_mid(self):
        assert find_ip_address("foo 1.2.3.4 bar") == [
            MatchResult(match="1.2.3.4", start=4, end=11)
        ]
        assert find_ip_address("foo 1:2:3:4:a:b:c:d bar") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=4, end=19)
        ]

    def test_single_match_end(self):
        assert find_ip_address("foo 1.2.3.4") == [
            MatchResult(match="1.2.3.4", start=4, end=11)
        ]
        assert find_ip_address("foo 1:2:3:4:a:b:c:d") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=4, end=19)
        ]

    def test_multi_match_start(self):
        assert find_ip_address("1.2.3.4 foo 1.2.3.4 bar") == [
            MatchResult(match="1.2.3.4", start=0, end=7),
            MatchResult(match="1.2.3.4", start=12, end=19),
        ]
        assert find_ip_address("1:2:3:4:a:b:c:d foo 1:2:3:4:a:b:c:d bar") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=0, end=15),
            MatchResult(match="1:2:3:4:a:b:c:d", start=20, end=35),
        ]

    def test_multi_match_mid(self):
        assert find_ip_address("bar 1.2.3.4 foo 1.2.3.4 bar") == [
            MatchResult(match="1.2.3.4", start=4, end=11),
            MatchResult(match="1.2.3.4", start=16, end=23),
        ]
        assert find_ip_address("bar 1:2:3:4:a:b:c:d foo 1:2:3:4:a:b:c:d bar") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=4, end=19),
            MatchResult(match="1:2:3:4:a:b:c:d", start=24, end=39),
        ]

    def test_multi_match_end(self):
        assert find_ip_address("bar 1.2.3.4 foo 1.2.3.4") == [
            MatchResult(match="1.2.3.4", start=4, end=11),
            MatchResult(match="1.2.3.4", start=16, end=23),
        ]
        assert find_ip_address("bar 1:2:3:4:a:b:c:d foo 1:2:3:4:a:b:c:d") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=4, end=19),
            MatchResult(match="1:2:3:4:a:b:c:d", start=24, end=39),
        ]

    def test_non_overlapping_match(self):
        assert find_ip_address("1.2.3.4,1:2:3:4:a:b:c:d") == [
            MatchResult(match="1.2.3.4", start=0, end=7),
            MatchResult(match="1:2:3:4:a:b:c:d", start=8, end=23),
        ]
        assert find_ip_address("1:2:3:4:a:b:c:d,1.2.3.4") == [
            MatchResult(match="1:2:3:4:a:b:c:d", start=0, end=15),
            MatchResult(match="1.2.3.4", start=16, end=23),
        ]


class TestReplaceIpAddress:
    def test_no_replacement(self):
        assert replace_ip_address("") == ""
        assert replace_ip_address("1.2") == "1.2"
        assert replace_ip_address("1.2.3.") == "1.2.3."
        assert replace_ip_address("256.1.2.3") == "256.1.2.3"
        assert replace_ip_address("g:h:i:j:k:l:m:n") == "g:h:i:j:k:l:m:n"

    def test_ipv4_address(self):
        assert replace_ip_address("255.1.2.3") == ""
        assert replace_ip_address("1.2.3.4") == ""
        assert replace_ip_address(".1.2.3.4.") == ".."
        assert replace_ip_address("a 1.2.3.4 b") == "a  b"

    def test_ipv6_address_case_1(self):
        assert replace_ip_address("1:2:3:4:5:6:7:8") == ""
        assert replace_ip_address("g1:2:3:4:5:6:7:8g") == "gg"

    def test_ipv6_address_case_2(self):
        assert replace_ip_address("1::") == ""
        assert replace_ip_address("1:2:3:4:5:6:7::") == ""
        assert replace_ip_address("g1:2:3:4:5:6:7::g") == "gg"

    def test_ipv6_address_case_3(self):
        assert replace_ip_address("1::8") == ""
        assert replace_ip_address("1:2:3:4:5:6::8") == ""
        assert replace_ip_address("g1:2:3:4:5:6::8g") == "gg"

    def test_ipv6_address_case_4(self):
        assert replace_ip_address("1::7:8") == ""
        assert replace_ip_address("1:2:3:4:5::7:8") == ""
        assert replace_ip_address("1:2:3:4:5::8") == ""
        assert replace_ip_address("g1:2:3:4:5::8g") == "gg"

    def test_ipv6_address_case_5(self):
        assert replace_ip_address("1::6:7:8") == ""
        assert replace_ip_address("1:2:3:4::6:7:8") == ""
        assert replace_ip_address("1:2:3:4::8") == ""
        assert replace_ip_address("g1:2:3:4::8g") == "gg"

    def test_ipv6_address_case_6(self):
        assert replace_ip_address("1::5:6:7:8") == ""
        assert replace_ip_address("1:2:3::5:6:7:8") == ""
        assert replace_ip_address("1:2:3::8") == ""
        assert replace_ip_address("g1:2:3::8g") == "gg"

    def test_ipv6_address_case_7(self):
        assert replace_ip_address("1::4:5:6:7:8") == ""
        assert replace_ip_address("1:2::4:5:6:7:8") == ""
        assert replace_ip_address("1:2::8") == ""
        assert replace_ip_address("g1:2::8g") == "gg"

    def test_ipv6_address_case_8(self):
        assert replace_ip_address("1::3:4:5:6:7:8") == ""
        assert replace_ip_address("g1::3:4:5:6:7:8g") == "gg"

    def test_ipv6_address_case_9(self):
        assert replace_ip_address("::2:3:4:5:6:7:8") == ""
        assert replace_ip_address("::8") == ""
        assert replace_ip_address("::") == ""
        assert replace_ip_address("g::8g") == "gg"

    def test_ipv6_address_case_10(self):
        # link-local IPv6 addresses with zone index
        assert replace_ip_address("fe80::7:8%eth0") == ""
        assert replace_ip_address("fe80::7:8%1") == ""
        assert replace_ip_address("gfe80::7:8%1g") == "g"

    def test_ipv6_address_case_11(self):
        # IPv4-mapped IPv6 addresses and IPv4-translated addresses
        assert replace_ip_address("::255.255.255.255") == ""
        assert replace_ip_address("::ffff:255.255.255.255") == ""
        assert replace_ip_address("g ::ffff:0:255.255.255.255 g") == "g  g"

    def test_ipv6_address_case_12(self):
        # IPv4-Embedded IPv6 Address
        assert replace_ip_address("2001:db8:3:4::192.0.2.33") == ""
        assert replace_ip_address("64:ff9b::192.0.2.33") == ""
        assert replace_ip_address("g 64:ff9b::192.0.2.33 g") == "g  g"


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
        assert find_email("foo3@bar4.co.foo1@bar2") == [
            MatchResult(match="foo3@bar4.co.foo1", start=0, end=17),
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


class TestFindPhoneNumber:

    def test_empty_string(self):
        assert find_phone_number("") == []

    def test_no_matches(self):
        assert find_phone_number("1") == []
        assert find_phone_number("654321") == []

    def test_whole_string_match(self):
        assert find_phone_number("18005551234") == [
            MatchResult(match="18005551234", start=0, end=11)
        ]
        assert find_phone_number("1 800 555 1234") == [
            MatchResult(match="1 800 555 1234", start=0, end=14)
        ]
        assert find_phone_number("+1 800 555-1234") == [
            MatchResult(match="+1 800 555-1234", start=0, end=15)
        ]
        assert find_phone_number("+86 800 555 1234") == [
            MatchResult(match="+86 800 555 1234", start=0, end=16)
        ]
        assert find_phone_number("1-800-555-1234") == [
            MatchResult(match="1-800-555-1234", start=0, end=14)
        ]
        assert find_phone_number("1 (800) 555-1234") == [
            MatchResult(match="1 (800) 555-1234", start=0, end=16)
        ]
        assert find_phone_number("(800)555-1234") == [
            MatchResult(match="(800)555-1234", start=0, end=13)
        ]
        assert find_phone_number("(800) 555-1234") == [
            MatchResult(match="(800) 555-1234", start=0, end=14)
        ]
        assert find_phone_number("(800)5551234") == [
            MatchResult(match="(800)5551234", start=0, end=12)
        ]
        assert find_phone_number("800-555-1234") == [
            MatchResult(match="800-555-1234", start=0, end=12)
        ]
        assert find_phone_number("800.555.1234") == [
            MatchResult(match="800.555.1234", start=0, end=12)
        ]
        assert find_phone_number("800 555 1234x5678") == [
            MatchResult(match="800 555 1234x5678", start=0, end=17)
        ]
        assert find_phone_number("8005551234 x5678") == [
            MatchResult(match="8005551234 x5678", start=0, end=16)
        ]
        assert find_phone_number("1    800    555-1234") == [
            MatchResult(match="1    800    555-1234", start=0, end=20)
        ]
        assert find_phone_number("1----800----555-1234") == [
            MatchResult(match="1----800----555-1234", start=0, end=20)
        ]

    def test_single_match_start(self):
        assert find_phone_number("18005551234bar") == [
            MatchResult(match="18005551234", start=0, end=11)
        ]
        assert find_phone_number("87654321bar") == [
            MatchResult(match="87654321", start=0, end=8)
        ]

    def test_single_match_mid(self):
        assert find_phone_number("foo18005551234bar") == [
            MatchResult(match="18005551234", start=3, end=14)
        ]
        assert find_phone_number("foo87654321bar") == [
            MatchResult(match="87654321", start=3, end=11)
        ]

    def test_single_match_end(self):
        assert find_phone_number("foo18005551234") == [
            MatchResult(match="18005551234", start=3, end=14)
        ]
        assert find_phone_number("foo87654321") == [
            MatchResult(match="87654321", start=3, end=11)
        ]

    def test_multi_match_start(self):
        assert find_phone_number("87654321 foo +86 800 555 1234 bar") == [
            MatchResult(match="87654321", start=0, end=8),
            MatchResult(match="+86 800 555 1234", start=13, end=29),
        ]

    def test_multi_match_mid(self):
        assert find_phone_number("bar87654321 foo +86 800 555 1234 bar") == [
            MatchResult(match="87654321", start=3, end=11),
            MatchResult(match="+86 800 555 1234", start=16, end=32),
        ]

    def test_multi_match_end(self):
        assert find_phone_number("bar87654321 foo +86 800 555 1234") == [
            MatchResult(match="87654321", start=3, end=11),
            MatchResult(match="+86 800 555 1234", start=16, end=32),
        ]

    def test_non_overlapping_match(self):
        assert find_phone_number("1-800-555-1234+86 800 555 1234") == [
            MatchResult(match="1-800-555-1234", start=0, end=14),
            MatchResult(match="+86 800 555 1234", start=14, end=30),
        ]
        assert find_phone_number("+86 800 555 12341-800-555-1234") == [
            MatchResult(match="+86 800 555 1234", start=0, end=16),
            MatchResult(match="1-800-555-1234", start=16, end=30),
        ]


class TestReplacePhoneNumber:
    def test_empty_string(self):
        assert replace_phone_number(s="", replacement="") == ""

    def test_no_matches(self):
        assert replace_phone_number(s="1", replacement="") == "1"
        assert replace_phone_number(s="654321", replacement="") == "654321"

    def test_whole_string_replaced(self):
        assert replace_phone_number(s="+86 800 555 1234", replacement="") == ""
        assert replace_phone_number(s="1-800-555-1234", replacement="") == ""

    def test_replacement(self):
        assert (
            replace_phone_number(
                s="foo1-800-555-1234bar+86 800 555 1234foo", replacement="x"
            )
            == "fooxbarxfoo"
        )


class TestFindSku:

    def test_empty_string(self):
        assert find_sku(s="") == []

    def test_no_matches(self):
        include = ".-"
        assert find_sku("f 1") == []
        assert find_sku("foo 123") == []
        assert find_sku("foo-123") == []
        assert find_sku("foo:123", include=include) == []

    def test_whole_string_match(self):
        include = ".-"
        assert find_sku("foo123") == [MatchResult(match="foo123", start=0, end=6)]
        assert find_sku(".foo123", include=include) == [
            MatchResult(match=".foo123", start=0, end=7)
        ]
        assert find_sku("foo.123", include=include) == [
            MatchResult(match="foo.123", start=0, end=7)
        ]
        assert find_sku("foo123.", include=include) == [
            MatchResult(match="foo123.", start=0, end=7)
        ]
        assert find_sku("-foo123", include=include) == [
            MatchResult(match="-foo123", start=0, end=7)
        ]
        assert find_sku("foo-123", include=include) == [
            MatchResult(match="foo-123", start=0, end=7)
        ]
        assert find_sku("foo123-", include=include) == [
            MatchResult(match="foo123-", start=0, end=7)
        ]

    def test_single_match_start(self):
        include = ".-"
        assert find_sku("foo123 bar 456") == [
            MatchResult(match="foo123", start=0, end=6)
        ]
        assert find_sku(".foo123 bar 456", include=include) == [
            MatchResult(match=".foo123", start=0, end=7)
        ]
        assert find_sku("foo.123 bar 456", include=include) == [
            MatchResult(match="foo.123", start=0, end=7)
        ]
        assert find_sku("foo123. bar 456", include=include) == [
            MatchResult(match="foo123.", start=0, end=7)
        ]
        assert find_sku("-foo123 bar 456", include=include) == [
            MatchResult(match="-foo123", start=0, end=7)
        ]
        assert find_sku("foo-123 bar 456", include=include) == [
            MatchResult(match="foo-123", start=0, end=7)
        ]
        assert find_sku("foo123- bar 456", include=include) == [
            MatchResult(match="foo123-", start=0, end=7)
        ]

    def test_single_match_mid(self):
        include = ".-"
        assert find_sku("bar foo123 456") == [
            MatchResult(match="foo123", start=4, end=10)
        ]
        assert find_sku("bar .foo123 456", include=include) == [
            MatchResult(match=".foo123", start=4, end=11)
        ]
        assert find_sku("bar foo.123 456", include=include) == [
            MatchResult(match="foo.123", start=4, end=11)
        ]
        assert find_sku("bar foo123. 456", include=include) == [
            MatchResult(match="foo123.", start=4, end=11)
        ]
        assert find_sku("bar -foo123 456", include=include) == [
            MatchResult(match="-foo123", start=4, end=11)
        ]
        assert find_sku("bar foo-123 456", include=include) == [
            MatchResult(match="foo-123", start=4, end=11)
        ]
        assert find_sku("bar foo123- 456", include=include) == [
            MatchResult(match="foo123-", start=4, end=11)
        ]

    def test_single_match_end(self):
        include = ".-"
        assert find_sku("bar 456 foo123") == [
            MatchResult(match="foo123", start=8, end=14)
        ]
        assert find_sku("bar 456 .foo123", include=include) == [
            MatchResult(match=".foo123", start=8, end=15)
        ]
        assert find_sku("bar 456 foo.123", include=include) == [
            MatchResult(match="foo.123", start=8, end=15)
        ]
        assert find_sku("bar 456 foo123.", include=include) == [
            MatchResult(match="foo123.", start=8, end=15)
        ]
        assert find_sku("bar 456 -foo123", include=include) == [
            MatchResult(match="-foo123", start=8, end=15)
        ]
        assert find_sku("bar 456 foo-123", include=include) == [
            MatchResult(match="foo-123", start=8, end=15)
        ]
        assert find_sku("bar 456 foo123-", include=include) == [
            MatchResult(match="foo123-", start=8, end=15)
        ]

    def test_multi_match_start(self):
        include = ".-"
        assert find_sku("foo123 foo123 bar 456") == [
            MatchResult(match="foo123", start=0, end=6),
            MatchResult(match="foo123", start=7, end=13),
        ]
        assert find_sku(".foo123 .foo123 bar 456", include=include) == [
            MatchResult(match=".foo123", start=0, end=7),
            MatchResult(match=".foo123", start=8, end=15),
        ]
        assert find_sku("foo.123 foo.123 bar 456", include=include) == [
            MatchResult(match="foo.123", start=0, end=7),
            MatchResult(match="foo.123", start=8, end=15),
        ]
        assert find_sku("foo123. foo123. bar 456", include=include) == [
            MatchResult(match="foo123.", start=0, end=7),
            MatchResult(match="foo123.", start=8, end=15),
        ]
        assert find_sku("-foo123 -foo123 bar 456", include=include) == [
            MatchResult(match="-foo123", start=0, end=7),
            MatchResult(match="-foo123", start=8, end=15),
        ]
        assert find_sku("foo-123 foo-123 bar 456", include=include) == [
            MatchResult(match="foo-123", start=0, end=7),
            MatchResult(match="foo-123", start=8, end=15),
        ]
        assert find_sku("foo123- foo123- bar 456", include=include) == [
            MatchResult(match="foo123-", start=0, end=7),
            MatchResult(match="foo123-", start=8, end=15),
        ]

    def test_multi_match_mid(self):
        include = ".-"
        assert find_sku("bar foo123 foo123 456") == [
            MatchResult(match="foo123", start=4, end=10),
            MatchResult(match="foo123", start=11, end=17),
        ]
        assert find_sku("bar .foo123 .foo123 456", include=include) == [
            MatchResult(match=".foo123", start=4, end=11),
            MatchResult(match=".foo123", start=12, end=19),
        ]
        assert find_sku("bar foo.123 foo.123 456", include=include) == [
            MatchResult(match="foo.123", start=4, end=11),
            MatchResult(match="foo.123", start=12, end=19),
        ]
        assert find_sku("bar foo123. foo123. 456", include=include) == [
            MatchResult(match="foo123.", start=4, end=11),
            MatchResult(match="foo123.", start=12, end=19),
        ]
        assert find_sku("bar -foo123 -foo123 456", include=include) == [
            MatchResult(match="-foo123", start=4, end=11),
            MatchResult(match="-foo123", start=12, end=19),
        ]
        assert find_sku("bar foo-123 foo-123 456", include=include) == [
            MatchResult(match="foo-123", start=4, end=11),
            MatchResult(match="foo-123", start=12, end=19),
        ]
        assert find_sku("bar foo123- foo123- 456", include=include) == [
            MatchResult(match="foo123-", start=4, end=11),
            MatchResult(match="foo123-", start=12, end=19),
        ]

    def test_multi_match_end(self):
        include = ".-"
        assert find_sku("bar 456 foo123 foo123") == [
            MatchResult(match="foo123", start=8, end=14),
            MatchResult(match="foo123", start=15, end=21),
        ]
        assert find_sku("bar 456 .foo123 .foo123", include=include) == [
            MatchResult(match=".foo123", start=8, end=15),
            MatchResult(match=".foo123", start=16, end=23),
        ]
        assert find_sku("bar 456 foo.123 foo.123", include=include) == [
            MatchResult(match="foo.123", start=8, end=15),
            MatchResult(match="foo.123", start=16, end=23),
        ]
        assert find_sku("bar 456 foo123. foo123.", include=include) == [
            MatchResult(match="foo123.", start=8, end=15),
            MatchResult(match="foo123.", start=16, end=23),
        ]
        assert find_sku("bar 456 -foo123 -foo123", include=include) == [
            MatchResult(match="-foo123", start=8, end=15),
            MatchResult(match="-foo123", start=16, end=23),
        ]
        assert find_sku("bar 456 foo-123 foo-123", include=include) == [
            MatchResult(match="foo-123", start=8, end=15),
            MatchResult(match="foo-123", start=16, end=23),
        ]
        assert find_sku("bar 456 foo123- foo123-", include=include) == [
            MatchResult(match="foo123-", start=8, end=15),
            MatchResult(match="foo123-", start=16, end=23),
        ]

    def test_non_overlapping_match(self):
        include = ".-"
        assert find_sku("foo123-bar456") == [
            MatchResult(match="foo123", start=0, end=6),
            MatchResult(match="bar456", start=7, end=13),
        ]
        assert find_sku("foo123,bar.456,car-789", include=include) == [
            MatchResult(match="foo123", start=0, end=6),
            MatchResult(match="bar.456", start=7, end=14),
            MatchResult(match="car-789", start=15, end=22),
        ]


class TestReplaceSku:
    def test_empty_string(self):
        assert replace_sku(s="", replacement="") == ""

    def test_no_matches(self):
        include = ".-"
        assert replace_sku("f 1", replacement="") == "f 1"
        assert replace_sku("foo 123", replacement="") == "foo 123"
        assert replace_sku("foo-123", replacement="") == "foo-123"
        assert replace_sku("foo:123", replacement="", include=include) == "foo:123"

    def test_whole_string_replaced(self):
        include = ".-"
        assert replace_sku("foo123", replacement="redacted") == "redacted"
        assert replace_sku(".foo123", replacement="", include=include) == ""
        assert replace_sku("foo.123", replacement="", include=include) == ""
        assert replace_sku("foo123.", replacement="", include=include) == ""
        assert replace_sku("-foo123", replacement="", include=include) == ""
        assert replace_sku("foo-123", replacement="", include=include) == ""
        assert replace_sku("foo123-", replacement="", include=include) == ""

    def test_multiple_replacements(self):
        include = ".-"
        assert (
            replace_sku(
                "foo123 foo123 bar foo123 foo123 456 foo123 foo123",
                replacement="redacted",
            )
            == "redacted redacted bar redacted redacted 456 redacted redacted"
        )
        assert (
            replace_sku(
                ".foo123 .foo123 bar .foo123 .foo123 456 .foo123 .foo123",
                include=include,
            )
            == "  bar   456  "
        )
        assert (
            replace_sku(
                "foo.123 foo.123 bar foo.123 foo.123 456 foo.123 foo.123",
                include=include,
            )
            == "  bar   456  "
        )
        assert (
            replace_sku(
                "foo123. foo123. bar foo123. foo123. 456 foo123. foo123.",
                include=include,
            )
            == "  bar   456  "
        )
        assert (
            replace_sku(
                "-foo123 -foo123 bar -foo123 -foo123 456 -foo123 -foo123",
                include=include,
            )
            == "  bar   456  "
        )
        assert (
            replace_sku(
                "foo-123 foo-123 bar foo-123 foo-123 456 foo-123 foo-123",
                include=include,
            )
            == "  bar   456  "
        )
        assert (
            replace_sku(
                "foo123- foo123- bar foo123- foo123- 456 foo123- foo123-",
                include=include,
            )
            == "  bar   456  "
        )
