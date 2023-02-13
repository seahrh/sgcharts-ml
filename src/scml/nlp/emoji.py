import importlib.resources
import re
from typing import List, NamedTuple, Tuple

import scml
from scml.nlp import split

__all__ = ["Emoji"]

log = scml.get_logger(__name__)


EMOJI_SHORTCODE_PATTERN = re.compile(r":([\w\s\-]+):", re.IGNORECASE)
CODE_POINT_PATTERN = re.compile(r"^[0-9A-F]{3,}", re.IGNORECASE)


class EmojiEntry(NamedTuple):
    codepoints: str
    status: str
    emoji: str
    name: str
    group: str
    subgroup: str


def to_unicode(base16: int) -> str:
    return r"\U" + hex(base16)[2:].zfill(8)


class Emoji:
    @staticmethod
    def _load() -> List[EmojiEntry]:
        res = []
        E_regex = re.compile(
            r" ?E\d+\.\d+ "
        )  # remove the pattern E<digit(s)>.<digit(s)>
        group = ""
        subgroup = ""
        from . import data

        with importlib.resources.open_text(data, "emoji-test.txt") as lines:
            for line in lines:  # skip the explanation lines
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith(
                    "#"
                ):  # these lines contain group and/or sub-group names
                    if "# group:" in line:
                        group = line.split(":")[-1].strip()
                    if "# subgroup:" in line:
                        subgroup = line.split(":")[-1].strip()
                    continue
                if (
                    group == "Component"
                ):  # skin tones, and hair types, skip, as mentioned above
                    continue
                if CODE_POINT_PATTERN.search(
                    line
                ):  # if the line starts with a hexadecimal number (an emoji code point)
                    # here we define all the elements that will go into emoji entries
                    codepoints = line.split(";")[
                        0
                    ].strip()  # in some cases it is one and in others multiple code points
                    status = (
                        line.split(";")[-1].split()[0].strip()
                    )  # status: fully-qualified, minimally-qualified, unqualified
                    if line[-1] == "#":
                        # The special case where the emoji is actually the hash sign "#". In this case manually assign the emoji
                        if "fully-qualified" in line:
                            emoji = "#️⃣"
                        else:
                            emoji = (
                                "#⃣"  # they look the same, but are actually different
                            )
                    else:  # the default case
                        emoji = (
                            line.split("#")[-1].split()[0].strip()
                        )  # the emoji character itself
                    if line[-1] == "#":  # (the special case)
                        name = "#"
                    else:  # extract the emoji name
                        split_hash = line.split("#")[1]
                        rm_capital_E = E_regex.split(split_hash)[1]
                        name = rm_capital_E
                    res.append(
                        EmojiEntry(
                            codepoints=codepoints,
                            status=status,
                            emoji=emoji,
                            name=name,
                            group=group,
                            subgroup=subgroup,
                        )
                    )
        return res

    def __init__(
        self,
    ):
        self.entries = self._load()
        single_codepoint: List[str] = []
        multi_codepoint: List[str] = []
        for cps in [entry.codepoints.split() for entry in self.entries]:
            # get hexadecimal number by left padding 8 zeros e.g. '\U0001F44D'
            hexs = [r"\U" + cp.zfill(8) for cp in cps]
            ls = single_codepoint
            if len(hexs) > 1:
                ls = multi_codepoint
            ls.append("".join(hexs))
        # sorting by length in decreasing order is extremely important as demonstrated above
        multi_codepoint.sort(key=len, reverse=True)
        # drop the first 2 chars "\U"
        indices = [int(x[2:], base=16) for x in single_codepoint]
        single_codepoint = []
        for _min, _max in scml.contiguous_ranges(indices):
            cp = to_unicode(_min)
            if _min != _max:
                cp += f"-{to_unicode(_max)}"
            single_codepoint.append(cp)
        all_codepoints = multi_codepoint + [r"[" + "".join(single_codepoint) + r"]"]
        self.emoji_pattern: re.Pattern = re.compile(
            "|".join(all_codepoints), re.IGNORECASE
        )

    def strip(self, s: str, replacement: str = "") -> str:
        return self.emoji_pattern.sub(replacement, s)

    @staticmethod
    def strip_shortcode(s: str, replacement: str = "") -> str:
        return EMOJI_SHORTCODE_PATTERN.sub(replacement, s)

    @staticmethod
    def shortcode_to_text(
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
                res[: m.start()]
                + prefix
                + separator.join(tokens)
                + suffix
                + res[m.end() :]
            )
        return res
