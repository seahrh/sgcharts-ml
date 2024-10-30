import codecs
from typing import AnyStr, Tuple
from unicodedata import normalize

from scml.nlp import MosesPunctNormalizer, to_str

__all__ = ["to_ascii", "cp1252_to_utf8", "decode_escaped_bytes"]


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


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end  # type: ignore


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end  # type: ignore


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def cp1252_to_utf8(s: str) -> str:
    """Convert char encoding from Windows cp1252 to utf-8.

    References
    - https://stackoverflow.com/questions/26324622/what-characters-do-not-directly-map-from-cp1252-to-utf-8
    - https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330
    - https://github.com/affjljoo3581/Feedback-Prize-Competition/blob/master/src/utils/data_utils.py
    """
    res = (
        s.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    return res


def decode_escaped_bytes(s: str, encoding="utf-8") -> str:
    """Convert escaped bytes in the \\xhh format to unicode characters."""
    return (
        s.encode("latin1")  # To bytes, required by 'unicode-escape'
        .decode("unicode-escape")  # Perform the actual octal-escaping decode
        .encode("latin1")  # 1:1 mapping back to bytes
        .decode(encoding)
    )
