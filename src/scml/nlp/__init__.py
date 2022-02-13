__all__ = [
    "to_str",
    "to_bytes",
    "strip_punctuation",
    "to_ascii",
    "is_number",
    "count_digit",
    "count_alpha",
    "count_upper",
    "count_space",
    "count_punctuation",
    "collapse_whitespace",
    "RepeatingCharacter",
    "RepeatingSubstring",
    "split",
    "decode_escaped_bytes",
    "ngrams",
    "sentences",
    "has_1a1d",
    "strip_xml",
    "strip_url",
    "strip_ip_address",
    "emoji_shortcode_to_text",
]


import re
import string
from typing import AnyStr, Iterable, List, Set, Tuple
from unicodedata import normalize


def to_str(bytes_or_str: AnyStr, encoding="utf-8") -> str:
    """Based on Effective Python Item 3:
    Know the difference between bytes str and unicode
    """
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode(encoding)
    # Instance of str
    return str(bytes_or_str)


def to_bytes(bytes_or_str: AnyStr, encoding="utf-8") -> bytes:
    """Based on Effective Python Item 3:
    Know the difference between bytes str and unicode
    """
    if isinstance(bytes_or_str, str):
        return bytes_or_str.encode(encoding)
    # Instance of bytes
    return bytes_or_str


PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")


def strip_punctuation(s: str, replacement: str = "") -> str:
    return PUNCTUATION_PATTERN.sub(replacement, s)


def _strip_punctuation_deprecated(s: str) -> str:
    """This uses the 3-argument version of str.maketrans with arguments (x, y, z) where 'x' and 'y'
    must be equal-length strings and characters in 'x' are replaced by characters in 'y'.
    'z' is a string (string.punctuation here)
    where each character in the string is mapped to None
    translator = str.maketrans('', '', string.punctuation)
    This is an alternative that creates a dictionary mapping
    of every character from string.punctuation to None (this will also work)
    Based on https://stackoverflow.com/a/34294398/519951
    """
    translator = str.maketrans(dict.fromkeys(string.punctuation))
    return s.translate(translator)


def is_number(s: str) -> bool:
    """Based on https://stackoverflow.com/a/40097699/519951
    :param s: string
    :return: True if string is a number
    """
    try:
        num = float(s)
        # check for "nan" floats
        return num == num  # or use `math.isnan(num)`
    except ValueError:
        return False


def count_digit(s: str) -> int:
    return sum(c.isdigit() for c in s)


def count_alpha(s: str) -> int:
    return sum(c.isalpha() for c in s)


def count_upper(s: str) -> int:
    return sum(c.isupper() for c in s)


def count_space(s: str) -> int:
    return sum(c.isspace() for c in s)


def count_punctuation(s: str) -> int:
    return sum(c in string.punctuation for c in s)


def split(delimiters: Iterable[str], s: str, maxsplit: int = 0) -> List[str]:
    """Split the string over an iterable of delimiters.
    Based on https://stackoverflow.com/a/13184791
    """
    pattern = "|".join(map(re.escape, delimiters))
    return re.split(pattern, s, maxsplit)


MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+")


def collapse_whitespace(s: str, replacement: str = " ") -> str:
    """Collapse multiple whitespace into a single space character. Converts "\\n\\r\\f\\t" to space character.
    Also trim the string of whitespace on both ends.
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(replacement, s).strip()


class RepeatingCharacter:
    """Collapse repeating letters into `max_repeat` length.
    Based on https://stackoverflow.com/a/1660739/519951
    """

    @staticmethod
    def count(s: str) -> int:
        """Count repeating characters except digits."""
        res = 0
        for i in range(1, len(s)):
            if s[i].isdigit() or s[i].isspace() or s[i] != s[i - 1]:
                continue
            res += 1
        return res

    def __init__(
        self, max_times: int = 2, letters: bool = True, punctuation: bool = True
    ):
        self.max_times = max_times
        if self.max_times < 2:
            raise ValueError("max_repeat must be greater than 1")
        if not letters and not punctuation:
            raise ValueError("At least 1 flag must be true (letters, punctuation)")
        chars = ""
        if letters:
            chars += "a-zA-Z"
        if punctuation:
            chars += re.escape(str(string.punctuation))
        self.pattern: re.Pattern = re.compile(f"([{chars}])\\1{{{str(max_times)}}}")

    def collapse(self, s: str) -> str:
        return str(self.pattern.sub(r"\1" * self.max_times, s))


class RepeatingSubstring:
    """Collapse repeating substring of `min_length`.
    Based on https://stackoverflow.com/a/33705982/519951
    """

    def __init__(
        self,
        min_length: int = 2,
        max_times: int = 1,
        letters: bool = True,
        punctuation: bool = True,
        whitespace: bool = True,
    ):
        if min_length < 2:
            raise ValueError("min_length must be greater than 1")
        self.max_times = max_times
        if self.max_times < 1:
            raise ValueError("max_times must be greater than 0")
        chars = ""
        if letters:
            chars += "a-z"
        if punctuation:
            chars += re.escape(str(string.punctuation))
        if whitespace:
            chars += r"\s"
        self.pattern: re.Pattern = re.compile(
            f"([{chars}]{{{min_length},}}?)\\1{{{str(max_times)},}}",
            re.IGNORECASE,
        )

    def collapse(self, s: str) -> str:
        return str(self.pattern.sub(r"\1" * self.max_times, s))

    def count(self, s: str) -> int:
        res = 0
        for m in self.pattern.finditer(s):
            if m[0] == "":
                continue
            res += m[0].count(m[1]) - 1
        return res

    def count_char(self, s: str) -> int:
        res = 0
        for m in self.pattern.finditer(s):
            if m[0] == "":
                continue
            res += len(m[1]) * (m[0].count(m[1]) - 1)
        return res


# (?<!...) is a negative look-behind assertion.
# (?<=...) is a positive look-behind assertion.
# No spacing after period.
# Salutations like Dr. Mr.
# Sentence boundaries like period, question mark, exclamation mark.
# Based on https://stackoverflow.com/a/25736082/519951
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s")


def sentences(s: str, maxsplit: int = 0) -> List[str]:
    return SENTENCE_BOUNDARY_PATTERN.split(s, maxsplit)


def decode_escaped_bytes(s: str, encoding="utf-8") -> str:
    """Convert escaped bytes in the \\xhh format to unicode characters."""
    return (
        s.encode("latin1")  # To bytes, required by 'unicode-escape'
        .decode("unicode-escape")  # Perform the actual octal-escaping decode
        .encode("latin1")  # 1:1 mapping back to bytes
        .decode(encoding)
    )


def ngrams(
    tokens: Iterable[str], n: int, skip: Set[str] = None
) -> List[Tuple[str, ...]]:
    ts = []
    for t in tokens:
        if skip is not None and t in skip:
            continue
        ts.append(t)
    # do not enter loop if n > len(tokens)
    res = [tuple(ts[i : i + n]) for i in range(len(ts) - n + 1)]
    return res


def has_1a1d(s: str, include: str = "") -> bool:
    """Returns True if the string has at least one letter and one digit.
    Useful for detecting product or model codes.
    """
    es = re.escape(include)
    # Positive lookahead: at least one digit AND at least one letter
    # Allow only chars inside the whitelist
    ps = r"(?=.*\d)(?=.*[A-Za-z])^[A-Za-z\d" + es + r"]+$"
    p = re.compile(ps)
    m = p.match(s)
    if m is None:
        return False
    return True


# first char in the angular bracket cannot be digit or whitespace
XML_PATTERN = re.compile(r"<[^\d\s][^>]*>", re.IGNORECASE)


def strip_xml(s: str, replacement: str = "") -> str:
    return XML_PATTERN.sub(replacement, s)


URL_PATTERN = re.compile(
    r"(https?://(?:www\.|(?!www))[a-zA-Z0-9]{3,}\.[^\s]{2,}|www\.[a-zA-Z0-9]{3,}\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
)


def strip_url(s: str, replacement: str = "") -> str:
    return URL_PATTERN.sub(replacement, s)


IPV4_SEGMENT = r"(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
IPV4_ADDRESS = f"\\b({IPV4_SEGMENT}\\.){{3}}{IPV4_SEGMENT}\\b"
IPV4_ADDRESS_PATTERN = re.compile(IPV4_ADDRESS)
IPV6_SEGMENT = r"[0-9a-fA-F]{1,4}"
p1 = f"({IPV6_SEGMENT}:){{7}}{IPV6_SEGMENT}"  # 1:2:3:4:5:6:7:8
p2 = f"({IPV6_SEGMENT}:){{1,7}}:"  # 1::  or 1:2:3:4:5:6:7::
p3 = f"({IPV6_SEGMENT}:){{1,6}}:{IPV6_SEGMENT}"  # 1::8  or 1:2:3:4:5:6::8
p4 = f"({IPV6_SEGMENT}:){{1,5}}(:{IPV6_SEGMENT}){{1,2}}"  # 1::7:8 or 1:2:3:4:5::7:8 or 1:2:3:4:5::8
p5 = f"({IPV6_SEGMENT}:){{1,4}}(:{IPV6_SEGMENT}){{1,3}}"  # 1::6:7:8  or 1:2:3:4::6:7:8 or 1:2:3:4::8
p6 = f"({IPV6_SEGMENT}:){{1,3}}(:{IPV6_SEGMENT}){{1,4}}"  # 1::5:6:7:8         1:2:3::5:6:7:8   1:2:3::8
p7 = f"({IPV6_SEGMENT}:){{1,2}}(:{IPV6_SEGMENT}){{1,5}}"  # 1::4:5:6:7:8       1:2::4:5:6:7:8   1:2::8
p8 = f"{IPV6_SEGMENT}:((:{IPV6_SEGMENT}){{1,6}})"  # 1::3:4:5:6:7:8     1::8
p9 = f":((:{IPV6_SEGMENT}){{1,7}}|:)"  # ::2:3:4:5:6:7:8   ::8   ::
# fe80::7:8%eth0 or fe80::7:8%1 (link-local IPv6 addresses with zone index)
p10 = f"[fF][eE]80:(:{IPV6_SEGMENT}){{0,4}}%[0-9a-zA-Z]+"
# ::255.255.255.255  ::ffff:255.255.255.255  ::ffff:0:255.255.255.255 (IPv4-mapped IPv6 addresses and IPv4-translated addresses)
p11 = f"::([fF]{{4}}(:0{{1,4}})?:)?{IPV4_ADDRESS}"
# 2001:db8:3:4::192.0.2.33  64:ff9b::192.0.2.33 (IPv4-Embedded IPv6 Address)
p12 = f"({IPV6_SEGMENT}:){{1,4}}:{IPV4_ADDRESS}"
# sort the patterns from more specific to less
IPV6_ADDRESS = f"({p12}|{p11}|{p10}|{p9}|{p8}|{p7}|{p6}|{p5}|{p4}|{p3}|{p2}|{p1})"
IPV6_ADDRESS_PATTERN = re.compile(IPV6_ADDRESS)


def strip_ip_address(s: str, replacement: str = "") -> str:
    res = s
    res = IPV6_ADDRESS_PATTERN.sub(replacement, res)
    res = IPV4_ADDRESS_PATTERN.sub(replacement, res)
    return res


EMOJI_SHORTCODE_PATTERN = re.compile(r":([\w\s\-]+):", re.IGNORECASE)


def emoji_shortcode_to_text(
    s: str,
    prefix: str = "(",
    suffix: str = ")",
    separator: str = " ",
    shortcode_separators: Tuple[str, ...] = ("\\s", "_", "-"),
) -> str:
    res = s
    for m in EMOJI_SHORTCODE_PATTERN.finditer(s):
        if len(m[1]) == 0:
            continue
        tokens = split(shortcode_separators, s=m[1])
        res = (
            res[: m.start()] + prefix + separator.join(tokens) + suffix + res[m.end() :]
        )
    return res


from .contractions import *

__all__ += contractions.__all__  # type: ignore  # module name is not defined

from .punctnorm import *

__all__ += punctnorm.__all__  # type: ignore  # module name is not defined

from .slang import *

__all__ += slang.__all__  # type: ignore  # module name is not defined

from .emoticons import *

__all__ += emoticons.__all__  # type: ignore  # module name is not defined


def to_ascii(s: AnyStr) -> str:
    """Normalise (normalize) unicode data in Python to remove umlauts, accents etc.
    Also converts curly quotes [] to straight quotes.
    Based on https://gist.github.com/j4mie/557354
    """
    # unicodedata normalize
    # The normal form KD (NFKD) will apply the compatibility decomposition
    # i.e. replace all compatibility characters with their equivalents.
    # @url https://docs.python.org/3/library/unicodedata.html
    res = to_str(s)
    res = MosesPunctNormalizer(pre_replace_unicode_punct=True).normalize(res)
    return to_str(normalize("NFKD", res).encode("ASCII", "ignore"))
