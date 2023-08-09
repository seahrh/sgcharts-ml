import csv
import re
from importlib.resources import files
from typing import Iterable, Optional, Tuple

import scml

__all__ = ["ContractionExpansion"]

log = scml.get_logger(__name__)


class ContractionExpansion:
    @staticmethod
    def _load() -> Tuple[Tuple[str, str], ...]:
        res = []
        from . import data

        with files(data).joinpath("contractions.tsv").open("r", encoding="utf-8") as f:
            rows = csv.reader(f, delimiter="\t")
            for row in rows:
                res.append((row[0].strip(), row[1].strip()))
        return tuple(res)

    def __init__(
        self,
        rules: Optional[Iterable[Tuple[str, str]]] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        rs = rules if rules is not None else self._load()
        self._rules = []
        for pattern, replacement in rs:
            # custom word boundary: add chars like '+'
            # negative lookbehind and lookahead
            # see https://stackoverflow.com/questions/14232931/custom-word-boundaries-in-regular-expression
            pattern = r"(?<![\w+])" + re.escape(pattern) + r"(?![\w+])"
            replacement = f"{prefix}{replacement}{suffix}"
            self._rules.append((re.compile(pattern, re.IGNORECASE), replacement))

    def apply(self, s: str) -> str:
        res = s
        for pattern, replacement in self._rules:
            res = pattern.sub(replacement, res)
        return res
