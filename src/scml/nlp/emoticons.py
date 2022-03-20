import csv
import importlib.resources
import re
from typing import Iterable, Tuple

import scml

__all__ = ["EmoticonToText"]

log = scml.get_logger(__name__)


class EmoticonToText:
    @staticmethod
    def _load() -> Tuple[Tuple[str, str], ...]:
        res = []
        from . import data

        with importlib.resources.open_text(data, "emoticons.tsv") as f:
            rows = csv.reader(f, delimiter="\t")
            for row in rows:
                res.append((row[0].strip(), row[1].strip()))
        return tuple(res)

    @staticmethod
    def _requires_word_boundary(pattern: str):
        if ":" in pattern:
            return True
        return False

    def __init__(
        self,
        rules: Iterable[Tuple[str, str]] = None,
        prefix: str = "[",
        suffix: str = "]",
    ):
        rs = rules if rules is not None else self._load()
        self._rules = []
        for pattern, replacement in rs:
            p = re.escape(pattern)
            # custom word boundary: add chars like '+'
            # negative lookbehind and lookahead
            # see https://stackoverflow.com/questions/14232931/custom-word-boundaries-in-regular-expression
            if self._requires_word_boundary(pattern):
                p = r"(?<![\w+])" + p + r"(?![\w+])"
            replacement = f"{prefix}{replacement}{suffix}"
            # pattern should be case sensitive!
            self._rules.append((re.compile(p), replacement))

    def apply(self, s: str) -> str:
        res = s
        for pattern, replacement in self._rules:
            res = pattern.sub(replacement, res)
        return res
