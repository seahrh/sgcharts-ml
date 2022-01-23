import re
import csv
import importlib.resources
import scml
from typing import Iterable, Tuple

__all__ = ["SlangExpansion"]

log = scml.get_logger()


def _load() -> Tuple[Tuple[str, str], ...]:
    res = []
    from . import data

    with importlib.resources.open_text(data, "slang.tsv") as f:
        rows = csv.reader(f, delimiter="\t")
        for row in rows:
            res.append((row[0].strip(), row[1].strip()))
    return tuple(res)


class SlangExpansion:
    def __init__(
        self,
        rules: Iterable[Tuple[str, str]] = None,
        prefix: str = "[",
        suffix: str = "]",
    ):
        rs = rules if rules is not None else _load()
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
