# hanzi_greek.py
# Greek-scheme (poutto_greek.py) -> Pinyin(with tone digits), syllables joined by '-'

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import unicodedata

from poutto_greek import (
    SMOOTH,
    ROUGH,
    BREATHING_ON_SECOND,
    PINYIN_FINAL_TO_GREEK,
    PINYIN_FINAL_TO_GREEK_RETRO,
)

GREEK_VOWELS = set("αεηιουωΑΕΗΙΟΥΩ")


# ----------------------------
# Breathing detection
# ----------------------------
def _strip_and_detect_breathing(word: str) -> Tuple[str, str]:
    """
    Returns (clean_word, breathing_type) where breathing_type in {"rough","smooth",""}.
    Removes smooth/rough marks from returned word.
    Supports breathing on second letter for initial vowel digraphs.
    """
    if not word:
        return word, ""

    w = unicodedata.normalize("NFD", word)

    base_letters = []
    base_positions = []
    for idx, ch in enumerate(w):
        if unicodedata.combining(ch) == 0:
            base_letters.append(ch)
            base_positions.append(idx)
            if len(base_letters) == 2:
                break

    breathing = ""
    if base_letters:
        first = base_letters[0]
        if first in GREEK_VOWELS:
            second = base_letters[1] if len(base_letters) >= 2 else ""
            digraph = (first + second).lower() if second else ""
            inspect_pos = base_positions[0]
            if len(base_letters) >= 2 and digraph in BREATHING_ON_SECOND:
                inspect_pos = base_positions[1]

            j = inspect_pos + 1
            while j < len(w) and unicodedata.combining(w[j]) != 0:
                if w[j] == ROUGH:
                    breathing = "rough"
                    break
                if w[j] == SMOOTH and not breathing:
                    breathing = "smooth"
                j += 1

    w_clean = w.replace(SMOOTH, "").replace(ROUGH, "")
    return unicodedata.normalize("NFC", w_clean), breathing


# ----------------------------
# Invert finals mapping from forward converter
# greek_final -> list[(pinyin_final, require_retro)]
# ----------------------------
FinalCand = Tuple[str, Optional[bool]]
FINAL_G2P: Dict[str, List[FinalCand]] = {}


def _add_final(surface: str, fin: str, require_retro: Optional[bool]):
    surface = unicodedata.normalize("NFC", surface)
    FINAL_G2P.setdefault(surface, [])
    item = (fin, require_retro)
    if item not in FINAL_G2P[surface]:
        FINAL_G2P[surface].append(item)


for fin_pin, surf in PINYIN_FINAL_TO_GREEK.items():
    _add_final(surf, fin_pin, None)

for fin_pin, surf in PINYIN_FINAL_TO_GREEK_RETRO.items():
    _add_final(surf, fin_pin, True)

# apical-i retroflex: pin_fin == "i" with zh/ch/sh/r maps to "ᾳ" in poutto_greek.py
_add_final("ᾳ", "i", True)

ALL_FIN = sorted(FINAL_G2P.keys(), key=len, reverse=True)


# ----------------------------
# Initials ambiguity (inverse)
# ----------------------------
INI_G2P_CAND: Dict[str, List[str]] = {
    "μ": ["m"],
    "ν": ["n"],
    "β": ["b"],
    "π": ["p"],
    "δ": ["d"],
    "τ": ["t"],
    "λ": ["l"],
    "ρ": ["r"],
    "φ": ["f"],
    "θ": ["s", "sh"],  # s and sh both -> θ
    "ψ": ["z", "zh"],  # z and zh both -> ψ
    "ζ": ["c", "ch"],  # c and ch both -> ζ
    "χ": ["h", "x"],  # h and x both -> χ
    "γ": ["g", "j"],  # g and j both -> γ
    "κ": ["k", "q"],  # k and q both -> κ
}
ALL_INI = sorted(INI_G2P_CAND.keys(), key=len, reverse=True)


# ----------------------------
# Tone suffix decoding (syllable-final) - MUST match poutto_greek.py
# ----------------------------
TONE_SUFFIXES: List[Tuple[str, int, bool]] = [
    ("λετο", 1, True),  # word-final tone1
    ("σαι", 2, True),  # word-final tone2
    ("λον", 1, True),  # word-medial tone1 before vowel-start
    ("σσ", 3, True),  # word-medial tone3 before vowel-start
    ("τος", 3, True),  # word-final tone3
    ("σο", 2, True),  # word-medial tone2 before consonant
    ("σ", 2, True),  # word-medial tone2 before vowel-start
    ("ξο", 4, True),  # word-medial tone4 before consonant
    ("ξ", 4, True),  # word-final OR before vowel-start
    ("ς", 5, True),  # neutral tone
    ("", 1, False),  # default: tone1, implicit
]

# gemination specials (match poutto_greek.py)
SPECIAL_GEM_PREFIX = {"φθ": "φ", "σθ": "θ", "χθ": "χ", "πτ": "ψ", "στ": "ζ", "κτ": "ξ"}
GEMMABLE_FIRST = set("μνβπδτλρφθχγκψζξ")


def _normalize_gem_prefix(s: str) -> Tuple[str, bool]:
    for pref, norm in SPECIAL_GEM_PREFIX.items():
        if s.startswith(pref):
            return norm + s[len(pref) :], True
    if len(s) >= 2 and s[0] == s[1]:
        return s[0] + s[2:], True
    return s, False


# ----------------------------
# vowel-start detection for suffix legality
# IMPORTANT: next syllable may start with "ρο" (er with zero-initial)
# ----------------------------
_VOWEL_START_LETTERS = set("αεηιουωᾳῃῳάέήίόύώῖῦῆῶύέάώ")


def _surface_vowel_start(word: str, pos: int) -> bool:
    if pos >= len(word):
        return False
    if word.startswith(
        "ρο", pos
    ):  # er encoded as ρο but still "zero-initial" in your design
        return True
    return word[pos] in _VOWEL_START_LETTERS


# ----------------------------
# Pinyin legality + spelling
# ----------------------------
def _is_front_final(fin: str) -> bool:
    return (
        fin.startswith("i")
        or fin.startswith("ü")
        or fin
        in {"in", "ing", "ian", "iang", "iao", "ie", "iu", "iong", "ün", "üan", "üe"}
    )


def _valid_combo(initial: str, final: str) -> bool:
    """
    Strong pruning to kill impossible outputs like sia/zia.
    Mirrors your latin reverse legality constraints.
    """
    retro = initial in {"zh", "ch", "sh", "r"}
    dental = initial in {"z", "c", "s"}

    # o/uo disambiguation:
    # 'o' only after b/p/m/f; other initials should use 'uo'; zero-initial never outputs bare 'o'
    if final == "o":
        return initial in {"b", "p", "m", "f"}
    if final == "uo":
        if initial in {"b", "p", "m", "f"}:
            return False

    # i-family restriction for dental/retro: only apical 'i' allowed
    if (retro or dental) and final.startswith("i") and final != "i":
        return False
    # ü-family forbidden after dental/retro
    if (retro or dental) and final.startswith("ü"):
        return False
    return True


def _spell_pinyin(initial: str, final: str) -> str:
    # contractions
    if final == "iou":
        final = "iu"

    if initial in {"j", "q", "x"} and final.startswith("ü"):
        final = "u" + final[1:]

    if initial:
        return initial + final

    # zero initial orthography (y/w)
    if final == "i":
        return "yi"
    if final == "in":
        return "yin"
    if final == "ing":
        return "ying"
    if final == "ie":
        return "ye"
    if final in {"iu", "iou"}:
        return "you"
    if final == "ia":
        return "ya"
    if final == "ian":
        return "yan"
    if final == "iang":
        return "yang"
    if final == "iao":
        return "yao"
    if final == "iong":
        return "yong"
    if final.startswith("i"):
        return "y" + final

    if final == "ü":
        return "yu"
    if final == "üe":
        return "yue"
    if final == "üan":
        return "yuan"
    if final == "ün":
        return "yun"
    if final.startswith("ü"):
        return "yu" + final[1:]

    if final == "u":
        return "wu"
    if final == "uo":
        return "wo"
    if final == "ua":
        return "wa"
    if final == "uai":
        return "wai"
    if final == "uan":
        return "wan"
    if final == "uang":
        return "wang"
    if final == "ui":
        return "wei"
    if final == "un":
        return "wen"
    if final == "ueng":
        return "weng"
    if final.startswith("u"):
        return "w" + final

    # forbid bare o as zero-initial in this system: prefer wo
    if final == "o":
        return "wo"

    return final


def _choose_initial(
    ini_surface: str, fin_pin: str, require_retro: Optional[bool]
) -> str:
    if ini_surface == "":
        return ""

    # merged initials: retro vs non-retro
    if ini_surface == "θ":
        return "sh" if require_retro is True else "s"
    if ini_surface == "ψ":
        return "zh" if require_retro is True else "z"
    if ini_surface == "ζ":
        return "ch" if require_retro is True else "c"

    # χ/γ/κ disambiguation by front finals
    front = _is_front_final(fin_pin)
    if ini_surface == "χ":
        return "x" if front else "h"
    if ini_surface == "γ":
        return "j" if front else "g"
    if ini_surface == "κ":
        return "q" if front else "k"

    return INI_G2P_CAND[ini_surface][0]


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Syllable:
    ini_g: str
    fin_g: str
    tone: int
    ini_pin: str
    fin_pin: str
    require_retro: Optional[bool]
    explicit_tone: bool
    geminated_ini: bool


# ----------------------------
# Parse one syllable at position
# ----------------------------
def _parse_syllable_at(word: str, pos: int) -> List[Tuple[int, Syllable]]:
    out: List[Tuple[int, Syllable]] = []

    ini_surfs: List[Tuple[str, bool]] = [("", False)]

    for ini in ALL_INI:
        if word.startswith(ini, pos):
            ini_surfs.append((ini, False))
        if ini and ini[0] in GEMMABLE_FIRST:
            doubled = ini[0] + ini
            if word.startswith(doubled, pos):
                ini_surfs.append((doubled, True))

    for pref in SPECIAL_GEM_PREFIX.keys():
        if word.startswith(pref, pos):
            ini_surfs.append((pref, True))

    for ini_surf, is_gem in ini_surfs:
        p1 = pos + len(ini_surf)

        for fin_g in ALL_FIN:
            if not word.startswith(fin_g, p1):
                continue
            p2 = p1 + len(fin_g)

            for suf, tone, explicit in TONE_SUFFIXES:
                if suf and not word.startswith(suf, p2):
                    continue

                next_pos = p2 + len(suf)

                # σ / lon only legal before vowel-start next syllable
                if suf in {"σ", "λον"}:
                    if next_pos >= len(word):
                        continue
                    if not _surface_vowel_start(word, next_pos):
                        continue

                # ξ is legal word-final OR before vowel-start
                if suf == "ξ" and next_pos < len(word):
                    # if not final, must be vowel-start next
                    if not _surface_vowel_start(word, next_pos):
                        continue

                p3 = next_pos

                ini_norm, gem2 = _normalize_gem_prefix(ini_surf)
                gem_flag = is_gem or gem2

                for fin_pin, req_retro in FINAL_G2P.get(fin_g, []):
                    ini_pin = _choose_initial(ini_norm, fin_pin, req_retro)

                    # ✅ crucial: prune impossible pinyin combos (fix sia/zia etc.)
                    if not _valid_combo(ini_pin, fin_pin):
                        continue
                    if ini_pin == "" and fin_pin == "o":
                        continue

                    out.append(
                        (
                            p3,
                            Syllable(
                                ini_g=ini_norm,
                                fin_g=fin_g,
                                tone=tone,
                                ini_pin=ini_pin,
                                fin_pin=fin_pin,
                                require_retro=req_retro,
                                explicit_tone=explicit,
                                geminated_ini=gem_flag,
                            ),
                        )
                    )

    return out


# ----------------------------
# Decode one word
# ----------------------------
_GREEK_UPPER_TO_LOWER = str.maketrans(
    {
        "Α": "α",
        "Β": "β",
        "Γ": "γ",
        "Δ": "δ",
        "Ε": "ε",
        "Ζ": "ζ",
        "Η": "η",
        "Θ": "θ",
        "Ι": "ι",
        "Κ": "κ",
        "Λ": "λ",
        "Μ": "μ",
        "Ν": "ν",
        "Ξ": "ξ",
        "Ο": "ο",
        "Π": "π",
        "Ρ": "ρ",
        "Σ": "σ",
        "Τ": "τ",
        "Υ": "υ",
        "Φ": "φ",
        "Χ": "χ",
        "Ψ": "ψ",
        "Ω": "ω",
    }
)


def greek_word_to_pinyin(word: str) -> Optional[str]:
    word, breathing = _strip_and_detect_breathing(word)
    word = unicodedata.normalize("NFC", word).translate(_GREEK_UPPER_TO_LOWER)
    word = re.sub(r"σ$", "ς", word)

    if not re.search(r"[\u0370-\u03FF\u1F00-\u1FFF]", word):
        return None

    n = len(word)
    INF = 10**9
    dp = [INF] * (n + 1)
    prev: List[Optional[Tuple[int, Syllable]]] = [None] * (n + 1)
    dp[0] = 0

    for i in range(n):
        if dp[i] >= INF:
            continue
        for j, syl in _parse_syllable_at(word, i):
            cost = dp[i] + 1
            if not syl.explicit_tone:
                cost += 0.2
            if syl.explicit_tone:
                cost -= 0.05
            # tiny preference: gemination tends to be meaningful (tone3 evidence)
            if syl.geminated_ini:
                cost -= 0.02
            if cost < dp[j]:
                dp[j] = cost
                prev[j] = (i, syl)

    if dp[n] >= INF or prev[n] is None:
        return None

    sylls: List[Syllable] = []
    cur = n
    while cur > 0:
        i, syl = prev[cur]
        sylls.append(syl)
        cur = i
    sylls.reverse()

    # breathing rough => initial h when word starts with vowel (zero initial)
    if breathing == "rough" and sylls:
        if sylls[0].ini_pin == "" and (not word.startswith("υ")):
            sylls[0].ini_pin = "h"

    # infer previous tone=3 when current syllable is geminated and previous not explicit
    for k in range(1, len(sylls)):
        if sylls[k].geminated_ini and (not sylls[k - 1].explicit_tone):
            sylls[k - 1].tone = 3

    out = []
    for syl in sylls:
        base = _spell_pinyin(syl.ini_pin, syl.fin_pin)
        out.append(f"{base}{syl.tone}")
    return "-".join(out)


# ----------------------------
# Convert full text (preserve whitespace/punct)
# ----------------------------
TOKEN_RE = re.compile(r"(\s+|[^\s]+)")


def greek_text_to_pinyin(text: str) -> str:
    out = []
    for tok in TOKEN_RE.findall(text):
        if tok.isspace():
            out.append(tok)
            continue

        conv = greek_word_to_pinyin(tok)
        if conv is not None:
            out.append(conv)
            continue

        # try stripping punctuation around
        m = re.fullmatch(
            r"([^A-Za-z0-9\u0370-\u03FF\u1F00-\u1FFF]+)?(.+?)([^A-Za-z0-9\u0370-\u03FF\u1F00-\u1FFF]+)?",
            tok,
        )
        if not m:
            out.append(tok)
            continue

        pre, core, suf = m.group(1) or "", m.group(2), m.group(3) or ""
        conv2 = greek_word_to_pinyin(core)
        out.append(pre + (conv2 if conv2 is not None else core) + suf)

    return "".join(out)


if __name__ == "__main__":
    import sys

    print(greek_text_to_pinyin(sys.stdin.read()))
