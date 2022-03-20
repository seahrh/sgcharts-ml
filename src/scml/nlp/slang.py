import csv
import importlib.resources
import re
from typing import Iterable, Tuple

import scml

__all__ = ["SlangExpansion"]

log = scml.get_logger(__name__)


class SlangExpansion:
    @staticmethod
    def _load() -> Tuple[Tuple[str, str], ...]:
        res = []
        from . import data

        with importlib.resources.open_text(data, "slang.tsv") as f:
            rows = csv.reader(f, delimiter="\t")
            for row in rows:
                res.append((row[0].strip(), row[1].strip()))
        return tuple(res)

    def __init__(
        self,
        rules: Iterable[Tuple[str, str]] = None,
        prefix: str = "[",
        suffix: str = "]",
        separator: str = "; ",
        keep_original_term: bool = False,
    ):
        rs = rules if rules is not None else self._load()
        self._rules = []
        for pattern, replacement in rs:
            # custom word boundary: add chars like '+'
            # negative lookbehind and lookahead
            # see https://stackoverflow.com/questions/14232931/custom-word-boundaries-in-regular-expression
            p = r"(?<![\w+])(" + re.escape(pattern) + r")(?![\w+])"
            r = f"{prefix}{replacement}{suffix}"
            if keep_original_term:
                r = f"{prefix}\\1{separator}{replacement}{suffix}"
            self._rules.append((re.compile(p, re.IGNORECASE), r))

    def apply(self, s: str) -> str:
        res = s
        for pattern, replacement in self._rules:
            res = pattern.sub(replacement, res)
        return res
