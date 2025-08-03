import re
from itertools import chain
from typing import List, Tuple

import scml

__all__ = ["MosesPunctNormalizer"]

log = scml.get_logger(__name__)


class MosesPunctNormalizer:
    """
    This is a Python port of the Moses punctuation normalizer from
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/normalize-punctuation.perl
    """

    EXTRA_WHITESPACE = [  # lines 21 - 30
        (r"\r", r""),
        (r"\(", r" ("),
        (r"\)", r") "),
        (r" +", r" "),
        (r"\) ([.!:?;,])", r")\g<1>"),
        (r"\( ", r"("),
        (r" \)", r")"),
        (r"(\d) %", r"\g<1>%"),
        (r" :", r":"),
        (r" ;", r";"),
    ]

    NORMALIZE_UNICODE_IF_NOT_PENN = [(r"`", r"'"), (r"''", r' " ')]  # lines 33 - 34

    NORMALIZE_UNICODE = [  # lines 37 - 50
        ("„", r'"'),
        ("“", r'"'),
        ("”", r'"'),
        ("–", r"-"),
        ("—", r" - "),
        (r" +", r" "),
        ("´", r"'"),
        ("([a-zA-Z])‘([a-zA-Z])", r"\g<1>'\g<2>"),
        ("([a-zA-Z])’([a-zA-Z])", r"\g<1>'\g<2>"),
        ("‘", r"'"),
        ("‚", r"'"),
        ("’", r"'"),
        (r"''", r'"'),
        ("´´", r'"'),
        ("…", r"..."),
    ]

    FRENCH_QUOTES = [  # lines 52 - 57
        ("\u00a0«\u00a0", r'"'),
        ("«\u00a0", r'"'),
        ("«", r'"'),
        ("\u00a0»\u00a0", r'"'),
        ("\u00a0»", r'"'),
        ("»", r'"'),
    ]

    HANDLE_PSEUDO_SPACES = [  # lines 59 - 67
        ("\u00a0%", r"%"),
        ("nº\u00a0", "nº "),
        ("\u00a0:", r":"),
        ("\u00a0ºC", " ºC"),
        ("\u00a0cm", r" cm"),
        ("\u00a0\\?", "?"),
        ("\u00a0\\!", "!"),
        ("\u00a0;", r";"),
        (",\u00a0", r", "),
        (r" +", r" "),
    ]

    EN_QUOTATION_FOLLOWED_BY_COMMA = [(r'"([,.]+)', r'\g<1>"')]

    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = [
        (r',"', r'",'),
        (r'(\.+)"(\s*[^<])', r'"\g<1>\g<2>'),  # don't fix period at end of sentence
    ]

    DE_ES_CZ_CS_FR = [
        ("(\\d)\u00a0(\\d)", r"\g<1>,\g<2>"),
    ]

    OTHER = [
        ("(\\d)\u00a0(\\d)", r"\g<1>.\g<2>"),
    ]

    # Regex substitutions from replace-unicode-punctuation.perl
    # https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    REPLACE_UNICODE_PUNCTUATION = [
        ("，", ","),
        (r"。\s*", ". "),
        ("、", ","),
        ("”", '"'),
        ("“", '"'),
        ("∶", ":"),
        ("：", ":"),
        ("？", "?"),
        ("《", '"'),
        ("》", '"'),
        ("）", ")"),
        ("！", "!"),
        ("（", "("),
        ("；", ";"),
        ("」", '"'),
        ("「", '"'),
        ("０", "0"),
        ("１", "1"),
        ("２", "2"),
        ("３", "3"),
        ("４", "4"),
        ("５", "5"),
        ("６", "6"),
        ("７", "7"),
        ("８", "8"),
        ("９", "9"),
        (r"．\s*", ". "),
        ("～", "~"),
        ("’", "'"),
        ("…", "..."),
        ("━", "-"),
        ("〈", "<"),
        ("〉", ">"),
        ("【", "["),
        ("】", "]"),
        ("％", "%"),
    ]

    def __init__(
        self,
        lang: str = "en",
        penn: bool = True,
        norm_quote_commas: bool = True,
        norm_numbers: bool = True,
        pre_replace_unicode_punct: bool = False,
    ):
        """
        :param lang: The two-letter language code.
        :type lang: str
        :param penn: Normalize Penn Treebank style quotations.
        :type penn: bool
        :param norm_quote_commas: Normalize quotations and commas
        :type norm_quote_commas: bool
        :param norm_numbers: Normalize numbers
        :type norm_numbers: bool
        """
        s = [
            self.EXTRA_WHITESPACE,
            self.NORMALIZE_UNICODE,
            self.FRENCH_QUOTES,
            self.HANDLE_PSEUDO_SPACES,
        ]
        if penn:  # Adds the penn substitutions after extra_whitespace regexes.
            s.insert(1, self.NORMALIZE_UNICODE_IF_NOT_PENN)
        if norm_quote_commas:
            if lang == "en":
                s.append(self.EN_QUOTATION_FOLLOWED_BY_COMMA)
            elif lang in ["de", "es", "fr"]:
                s.append(self.DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA)
        if norm_numbers:
            if lang in ["de", "es", "cz", "cs", "fr"]:
                s.append(self.DE_ES_CZ_CS_FR)
            else:
                s.append(self.OTHER)
        self.substitutions: List[Tuple[str, str]] = list(chain(*s))
        self.pre_replace_unicode_punct = pre_replace_unicode_punct

    def replace_unicode_punct(self, text):
        for regexp, substitution in self.REPLACE_UNICODE_PUNCTUATION:
            text = re.sub(regexp, substitution, text)
        return text

    def normalize(self, text):
        """
        Returns a string with normalized punctuation.
        """
        # Optionally, replace unicode puncts BEFORE normalization.
        if self.pre_replace_unicode_punct:
            text = self.replace_unicode_punct(text)
        # Actual normalization.
        for regexp, substitution in self.substitutions:
            log.debug(regexp, substitution)
            text = re.sub(regexp, substitution, text)
            log.debug(text)
        return text.strip()
