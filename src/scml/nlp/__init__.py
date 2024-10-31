__all__ = [
    "to_str",
    "to_bytes",
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
    "ngrams",
    "sentences",
    "has_1a1d",
]


import re
import string
from typing import AnyStr, Iterable, List, Optional, Set, Tuple


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


def ngrams(
    tokens: Iterable[str], n: int, skip: Optional[Set[str]] = None
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


# To avoid circular imports, the order of module imports matters!

from .contractions import *

__all__ += contractions.__all__  # type: ignore  # module name is not defined

from .punctnorm import *

__all__ += punctnorm.__all__  # type: ignore  # module name is not defined

from .slang import *

__all__ += slang.__all__  # type: ignore  # module name is not defined

from .emoticons import *

__all__ += emoticons.__all__  # type: ignore  # module name is not defined

from .emoji import *

__all__ += emoji.__all__  # type: ignore  # module name is not defined

from .charencoding import *

__all__ += charencoding.__all__  # type: ignore  # module name is not defined

from .findreplace import *

__all__ += findreplace.__all__  # type: ignore  # module name is not defined

from .word2vec import *

__all__ += word2vec.__all__  # type: ignore  # module name is not defined
