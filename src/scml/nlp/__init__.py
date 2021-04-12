__all__ = [
    "to_str",
    "to_bytes",
    "strip_punctuation",
    "to_ascii_str",
    "is_number",
    "count_digit",
    "count_alpha",
    "count_upper",
    "count_space",
    "count_punctuation",
    "split",
]

import string
import re
from unicodedata import normalize
from typing import AnyStr, Iterable, Callable, List


def to_str(bytes_or_str: AnyStr, encoding="utf-8") -> str:
    """Based on Effective Python Item 3:
    Know the difference between bytes str and unicode
    """
    if isinstance(bytes_or_str, bytes):
        return bytes_or_str.decode(encoding)
    # Instance of str
    return bytes_or_str


def to_bytes(bytes_or_str: AnyStr, encoding="utf-8") -> bytes:
    """Based on Effective Python Item 3:
    Know the difference between bytes str and unicode
    """
    if isinstance(bytes_or_str, str):
        return bytes_or_str.encode(encoding)
    # Instance of bytes
    return bytes_or_str


def strip_punctuation(s: str) -> str:
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


def _to_ascii_str_transform_fn(s: str) -> str:
    res = s.replace("‘", "'")  # opening single quote
    res = res.replace("’", "'")  # closing single quote
    res = res.replace("“", '"')  # opening double quote
    res = res.replace("”", '"')  # closing double quote
    return res


def to_ascii_str(s: AnyStr, transform_fn: Callable = _to_ascii_str_transform_fn) -> str:
    """Normalise (normalize) unicode data in Python to remove umlauts, accents etc.
    Also converts curly quotes [] to straight quotes.
    Based on https://gist.github.com/j4mie/557354
    """
    # unicodedata normalize
    # The normal form KD (NFKD) will apply the compatibility decomposition
    # i.e. replace all compatibility characters with their equivalents.
    # @url https://docs.python.org/3/library/unicodedata.html
    res = to_str(s)
    res = transform_fn(res)
    res = to_str(normalize("NFKD", res).encode("ASCII", "ignore"))
    return res


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
    n = 0
    for c in s:
        if c.isdigit():
            n += 1
    return n


def count_alpha(s: str) -> int:
    n = 0
    for c in s:
        if c.isalpha():
            n += 1
    return n


def count_upper(s: str) -> int:
    n = 0
    for c in s:
        if c.isupper():
            n += 1
    return n


def count_space(s: str) -> int:
    n = 0
    for c in s:
        if c.isspace():
            n += 1
    return n


def count_punctuation(s: str) -> int:
    n = 0
    for c in s:
        if c in string.punctuation:
            n += 1
    return n


def split(delimiters: Iterable[str], s, maxsplit=0) -> List[str]:
    """Split the string over an iterable of delimiters.
    Based on https://stackoverflow.com/a/13184791
    """
    pattern = "|".join(map(re.escape, delimiters))
    return re.split(pattern, s, maxsplit)
