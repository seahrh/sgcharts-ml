__all__ = [
    "MatchResult",
    "find_non_overlapping",
    "find_email",
    "replace_email",
    "replace_punctuation",
    "replace_symbol",
    "replace_spans",
    "find_xml",
    "replace_xml",
    "find_phone_number",
    "replace_phone_number",
    "find_url",
    "replace_url",
    "find_ip_address",
    "replace_ip_address",
    "find_alphanumeric_id",
    "replace_alphanumeric_id",
]


import re
import string
from typing import List, NamedTuple, Optional, Sequence, Tuple


class MatchResult(NamedTuple):
    match: str
    start: int
    end: int


def find_non_overlapping(pattern: re.Pattern, s: str) -> List[MatchResult]:
    res: List[MatchResult] = []
    for m in pattern.finditer(s):
        if m[0] == "":
            continue
        res.append(MatchResult(match=m[0], start=m.start(0), end=m.end(0)))
    return res


# Based on RFC 5322 https://stackoverflow.com/a/201378/519951
EMAIL_PATTERN = re.compile(
    r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])",
    re.IGNORECASE,
)


def find_email(s: str) -> List[MatchResult]:
    return find_non_overlapping(pattern=EMAIL_PATTERN, s=s)


def replace_email(s: str, replacement: str = "") -> str:
    return EMAIL_PATTERN.sub(replacement, s)


PUNCTUATION_PATTERN = re.compile(f"[{re.escape(string.punctuation)}]")


def replace_punctuation(s: str, replacement: str = "") -> str:
    return PUNCTUATION_PATTERN.sub(replacement, s)


SYMBOL_STRING = "⌔⟐◇◆◈⬖⬗⬘⬙⬠⬡⎔◊⧫⬢⬣⋄▰▪◼▮◾▗▖■∎▃▄▅▆▇█▌▐▍▎▉▊▋❘❙❚▀▘▝▙▚▛▜▟▞░▒▓▂▁▬▔▫▯▭▱◽□◻▢⊞⊡⊟⊠▣▤▥▦⬚▧▨▩⬓◧⬒◨◩◪⬔⬕❏❐❑❒⧈◰◱◳◲◫⧇⧅⧄⍁⍂⎔⎕⏣⌓⏥⏢⊞⊟⊠⊡▲▼◀▶←↑→↓↔↕∞±×÷≠≥≤♀♂★☆♠♦♣♥♡■□●○◆◇✖✚✔♫✈⚑❮❯©®㊤㊦㊧㊨㊥㊣㊖㊕㍿αβπθ¥€【】︻︼⸨⸩❪❫⏠⏡⌌⌍⌎⌏⌐⌙⌢⌣⎴⎵⎶⎾⎿⏋⏌¬❨❩⸨⸩◖◗❪❫❮❯❬❭❰❱⊏⊐⊑⊒⟦⟧⟨⟩⟪⟫⦃⦄⦅⦆⦇⦈⦉⦊⦋⦌⦍⦎⦏⦐⦑⦒⦓⦔⦕⦖⦗⦘❬❭❮❯❰❱❴❵❲❳⦗⦘⁅⁆〈〉⏜⏝⏞⏟{}⸨⸩❨❩❪❫⸦⸧⸡⸠⸢⸣⸤⸥⎡⎤⎣⎦⎨⎬⌠⌡⎛⎠⎝⎞⁀⁔‿⁐⎰⎱◜◝◞◟◠◡⋒⋓⋐⋑╰╮╭╯⌞⌟⌜⌝⌊⌋⌉⌈⌋▲▼◀▶◢◣◥◤△▽◿◺◹◸▴▾◂▸▵▿◃▹◁▷◅▻◬⟁⧋⧊⊿∆∇◭◮⧩⧨⦉⦊►◄⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾⓪①②③④⑤⑥⑦⑧⑨⑩➀➁➂➃➄➅➆➇➈➉⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⓿❶❷❸❹❺❻❼❽❾❿➊➋➌➍➎➏➐➑➒➓⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴⚬○⚪◌◍◎◯❍◉⦾⊙⦿⊜⊖⊘⊚⊛⊝●⚫⦁◐◑◒◓◔◕⦶⦸◵◴◶◷⊕⊗•●⦇⦈⊕⊖⊗⊘⊙⊚⊛⊜⊝■。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞◎＠※◆●［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
SYMBOL_PATTERN = re.compile(f"[{re.escape(SYMBOL_STRING)}]")


def replace_symbol(s: str, replacement: str = "") -> str:
    return SYMBOL_PATTERN.sub(replacement, s)


def replace_spans(
    s: str,
    positions: Sequence[Tuple[int, int]],
    replacements: Optional[Sequence[str]] = None,
) -> str:
    if replacements is not None and len(replacements) != len(positions):
        raise ValueError("Length of replacements and positions must be equal")
    keep: List[bool] = [True] * len(s)
    for p in positions:
        i = 0
        # starting index must not be negative
        while 0 <= p[0] + i < p[1]:
            keep[p[0] + i] = False
            i += 1
    res: str = ""
    i, j = 0, 0
    while i < len(s):
        steps = 1
        if keep[i]:
            res += s[i]
        elif replacements is not None:
            if 0 <= positions[j][0] < positions[j][1]:  # valid span
                res += replacements[j]
                steps = positions[j][1] - positions[j][0]
            j += 1
        i += steps
    return res


# first char enclosed inside the angular brackets cannot be digit or whitespace
XML_PATTERN = re.compile(r"(<[^\d\s][^>]*>)", re.IGNORECASE)


def find_xml(s: str) -> List[MatchResult]:
    return find_non_overlapping(pattern=XML_PATTERN, s=s)


def replace_xml(s: str, replacement: str = "") -> str:
    return XML_PATTERN.sub(replacement, s)


# Phone numbers that are at least 7 digits long
# Based on https://stackoverflow.com/a/16702965/519951
PHONE_NUMBER_PATTERN = re.compile(
    r"(?:\+?(\d{1,3}))?[-. (]*(\d{0,3})[-. )]*(\d{3,4})[-. ]*(\d{4})(?: *x(\d+))?",
    re.IGNORECASE,
)


def find_phone_number(s: str) -> List[MatchResult]:
    return find_non_overlapping(pattern=PHONE_NUMBER_PATTERN, s=s)


def replace_phone_number(s: str, replacement: str = "") -> str:
    return PHONE_NUMBER_PATTERN.sub(replacement, s)


URL_PATTERN = re.compile(
    r"(https?://(?:www\.|(?!www))[a-zA-Z0-9]{3,}\.[^\s]{2,}|www\.[a-zA-Z0-9]{3,}\.[^\s]{2,}|https?://(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
    re.IGNORECASE,
)


def find_url(s: str) -> List[MatchResult]:
    return find_non_overlapping(pattern=URL_PATTERN, s=s)


def replace_url(s: str, replacement: str = "") -> str:
    return URL_PATTERN.sub(replacement, s)


IPV4_SEGMENT = r"(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])"
IPV4_ADDRESS = f"(\\b({IPV4_SEGMENT}\\.){{3}}{IPV4_SEGMENT}\\b)"
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


def find_ip_address(s: str) -> List[MatchResult]:
    res = find_non_overlapping(
        pattern=IPV6_ADDRESS_PATTERN, s=s
    ) + find_non_overlapping(pattern=IPV4_ADDRESS_PATTERN, s=s)
    res.sort(key=lambda x: x.start)
    return res


def replace_ip_address(s: str, replacement: str = "") -> str:
    res = s
    res = IPV6_ADDRESS_PATTERN.sub(replacement, res)
    res = IPV4_ADDRESS_PATTERN.sub(replacement, res)
    return res


# Alphanumeric identifier is defined as having:
# at least 1 letter AND 1 digit, plus optionally a character whitelist.
# first positive lookahead "(?=[a-z]*\d)": any sequence of letters then a digit
# second positive lookahead "(?=\d*[a-z])": any sequence of digits then a letter
# capturing group "([a-z\d]{2,})": the sku must contain at least 2 whitelisted chars.
ALPHANUMERIC_ID_PATTERN = re.compile(
    r"(?=[a-z]*\d)(?=\d*[a-z])([a-z\d]{2,})", re.IGNORECASE
)


def _alphanumeric_id_pattern(include: str = "") -> re.Pattern:
    res = ALPHANUMERIC_ID_PATTERN
    if len(include) != 0:
        white = re.escape(include)
        res = re.compile(
            r"(?=[a-z"
            + white
            + r"]*\d)(?=[\d"
            + white
            + r"]*[a-z])([a-z\d"
            + white
            + r"]{2,})",
            re.IGNORECASE,
        )
    return res


def find_alphanumeric_id(s: str, include: str = "") -> List[MatchResult]:
    return find_non_overlapping(pattern=_alphanumeric_id_pattern(include=include), s=s)


def replace_alphanumeric_id(
    s: str,
    replacement: str = "",
    include: str = "",
) -> str:
    return _alphanumeric_id_pattern(include=include).sub(replacement, s)
