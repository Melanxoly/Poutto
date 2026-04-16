# hanzi.py
# Latin-scheme (poutto.py) -> Pinyin(with tone digits), syllables joined by '-'
#
# STRICTLY aligned with your current poutto.py:
#   - finals table including retro "false diphthongs/nasals"
#   - tone markers: 2nd {st,s,so}, 3rd {ss,t,ggo,nno}+gemination, 4th {v,vo}, neutral {s}
#   - tone-anchor insertion rule (same vowel set as poutto.py, notably excluding 'ú')
#   - dropped 'h' for zero-initial syllables after 2/3/4 in non-initial position
#
# Implementation: precomputed surface->candidates + DP to decode each token word.

import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

# ----------------------------
# Vowels used by poutto.py's tone-anchor logic (MUST match poutto.py _VOWELS)
# ----------------------------
VOWELS = set("aeiouéêùèà")
ASCII_LETTERS = set("abcdefghijklmnopqrstuvwxyz")
GEMMABLE_FIRST = set("mnbpdtgclrc")  # could be doubled by 3rd-tone gemination

# ----------------------------
# Inverse finals mapping (match current poutto.py tables)
# pinyin_final -> transcription_final
# ----------------------------
_FINAL_MAP_STD = {
    # single
    "i": "i",
    "ü": "u",  # transcription uses 'u' for ü in your scheme
    "u": "ou",
    "e": "e",
    "o": "uo",
    "a": "a",
    # diphthongs
    "ie": "é",
    "iou": "iu",
    "iu": "iu",
    "ia": "ia",
    "iao": "io",
    "üe": "ue",
    "ue": "ue",
    "ui": "oi",
    "uo": "uo",
    "ua": "oa",
    "uai": "oe",
    "ou": "eo",
    "ei": "ê",
    "ai": "ae",
    "ao": "au",
    # nasal
    "in": "ino",
    "ing": "igo",
    "ian": "iano",
    "iang": "iago",
    "ün": "uno",
    "un": "ouno",
    "üan": "uano",
    "uan": "oano",
    "uang": "oago",
    "iong": "ugo",
    "ueng": "ougo",
    "ong": "ogo",
    "en": "eno",
    "eng": "ego",
    "an": "ano",
    "ang": "ago",
    "er": "ir",
}

_FINAL_MAP_RETRO = {
    # false diphthongs (only for retroflex initials zh/ch/sh/r)
    "u": "ú",
    "e": "è",
    "a": "à",
    "ui": "ui",
    "uo": "eu",
    "ua": "ua",
    "uai": "uae",
    "ou": "io",
    "ei": "ie",
    "ai": "ia",
    "ao": "eau",
    # false nasal
    "un": "uno",
    "uan": "uano",
    "uang": "uago",
    "ueng": "uego",
    "ong": "uogo",
    "en": "ino",
    "eng": "igo",
    "an": "iano",
    "ang": "iago",
}

RETRO_INI_SURF = {
    "pr",
    "tr",
    "cr",
    "r",
}  # transcription initials that can represent retroflex family

# Build: trans_final -> list of (pinyin_final, require_retro)
FinalCand = Tuple[str, Optional[bool]]
FINAL_T2P: Dict[str, List[FinalCand]] = defaultdict(list)

for fin_pin, fin_tr in _FINAL_MAP_STD.items():
    FINAL_T2P[fin_tr].append((fin_pin, None))

for fin_pin, fin_tr in _FINAL_MAP_RETRO.items():
    FINAL_T2P[fin_tr].append((fin_pin, True))

# Special: retroflex apical "i" (zhi/chi/shi/ri) -> trans final 'é'
FINAL_T2P["é"].append(("i", True))

# De-dup
for k in list(FINAL_T2P.keys()):
    seen = set()
    uniq = []
    for cand in FINAL_T2P[k]:
        if cand in seen:
            continue
        seen.add(cand)
        uniq.append(cand)
    FINAL_T2P[k] = uniq

ALL_TRANS_FINALS = sorted(FINAL_T2P.keys(), key=len, reverse=True)

# ----------------------------
# Transcription initials -> candidate pinyin initials
# (match poutto.py _map_initial)
# ----------------------------
INI_T2P_CAND: Dict[str, List[str]] = {
    "": [""],  # dropped h => zero initial
    "h": [""],  # explicit zero-initial marker
    "m": ["m"],
    "n": ["n"],
    "b": ["b"],
    "p": ["p"],
    "d": ["d"],
    "t": ["t"],
    "l": ["l"],
    "r": ["r"],
    "ph": ["f"],
    "g": ["g", "j"],  # g/j merged
    "c": ["k", "q"],  # k/q merged
    "ch": ["h", "x"],  # h/x merged
    "pr": ["z", "zh"],  # z/zh merged
    "tr": ["c", "ch"],  # c/ch merged
    "cr": ["s", "sh"],  # s/sh merged
}

TRANS_INITIALS = sorted(INI_T2P_CAND.keys(), key=len, reverse=True)


# ----------------------------
# Tone anchor (must match poutto.py)
# ----------------------------
def _tone_anchor_pos(trans: str) -> int:
    cut = len(trans)
    if trans.endswith(("no", "go")):
        cut -= 2
    elif trans.endswith("r"):
        cut -= 1
    idx = -1
    for i, ch in enumerate(trans[:cut]):
        if ch in VOWELS:
            idx = i
    return idx


def _insert_after_anchor(trans: str, ins: str) -> str:
    idx = _tone_anchor_pos(trans)
    if idx < 0:
        return trans + ins
    return trans[: idx + 1] + ins + trans[idx + 1 :]


def _is_vowel_start_char(ch: str) -> bool:
    return ch in VOWELS


def _looks_vowel_start(word: str, pos: int) -> bool:
    if pos >= len(word):
        return False
    ch = word[pos]
    return ch == "h" or _is_vowel_start_char(ch)


# ----------------------------
# Candidate syllable record
# ----------------------------
@dataclass(frozen=True)
class Cand:
    ini_surf: str
    fin_tr: str
    fin_pin: str
    require_retro: Optional[bool]  # True or None
    tone: int  # 1..5
    explicit_tone: bool
    geminated_ini: bool
    marker: str


# ----------------------------
# Precompute surface forms -> candidates
# ----------------------------
SURFACE2CANDS: Dict[str, List[Cand]] = defaultdict(list)


def _add_surface(surface: str, cand: Cand) -> None:
    SURFACE2CANDS[surface].append(cand)


def _is_consonant_start(surface: str) -> bool:
    if not surface:
        return False
    ch = surface[0]
    if ch in VOWELS:
        return False
    if ch == "h":
        return False
    return ch in ASCII_LETTERS


def _build_surfaces() -> None:
    for ini in TRANS_INITIALS:
        for fin_tr in ALL_TRANS_FINALS:
            for fin_pin, req_retro in FINAL_T2P[fin_tr]:
                if req_retro is True and ini not in RETRO_INI_SURF:
                    continue

                base = ini + fin_tr

                # 1) tone=1 (no marker)
                _add_surface(
                    base, Cand(ini, fin_tr, fin_pin, req_retro, 1, False, False, "none")
                )

                # 2) tone=5 (neutral): append 's'
                _add_surface(
                    base + "s",
                    Cand(ini, fin_tr, fin_pin, req_retro, 5, True, False, "light_s"),
                )

                # 3) tone=2: st / s / so
                for mk, ins in (("st", "st"), ("s", "s"), ("so", "so")):
                    surf = _insert_after_anchor(base, ins)
                    _add_surface(
                        surf, Cand(ini, fin_tr, fin_pin, req_retro, 2, True, False, mk)
                    )

                # 4) tone=3: ss / t + word-final nasal ggo/nno
                surf_ss = _insert_after_anchor(base, "ss")
                _add_surface(
                    surf_ss, Cand(ini, fin_tr, fin_pin, req_retro, 3, True, False, "ss")
                )

                surf_t = _insert_after_anchor(base, "t")
                _add_surface(
                    surf_t, Cand(ini, fin_tr, fin_pin, req_retro, 3, True, False, "t")
                )

                if base.endswith("go"):
                    _add_surface(
                        base[:-2] + "ggo",
                        Cand(ini, fin_tr, fin_pin, req_retro, 3, True, False, "ggo"),
                    )
                if base.endswith("no"):
                    _add_surface(
                        base[:-2] + "nno",
                        Cand(ini, fin_tr, fin_pin, req_retro, 3, True, False, "nno"),
                    )

                # 5) tone=4: v / vo
                for mk, ins in (("v", "v"), ("vo", "vo")):
                    surf = _insert_after_anchor(base, ins)
                    _add_surface(
                        surf, Cand(ini, fin_tr, fin_pin, req_retro, 4, True, False, mk)
                    )

    # geminated variants: first letter doubled (used to infer previous tone=3 before consonant)
    extra: Dict[str, List[Cand]] = defaultdict(list)
    for surf, cands in SURFACE2CANDS.items():
        if not surf or not _is_consonant_start(surf):
            continue
        first = surf[0]
        if first not in GEMMABLE_FIRST:
            continue
        gem = first + surf
        for c in cands:
            extra[gem].append(
                Cand(
                    c.ini_surf,
                    c.fin_tr,
                    c.fin_pin,
                    c.require_retro,
                    c.tone,
                    c.explicit_tone,
                    True,
                    c.marker,
                )
            )
    for k, v in extra.items():
        SURFACE2CANDS[k].extend(v)


_build_surfaces()

PATTERNS_BY_FIRST: DefaultDict[str, List[Tuple[str, Cand]]] = defaultdict(list)
for surf, cands in SURFACE2CANDS.items():
    if not surf:
        continue
    PATTERNS_BY_FIRST[surf[0]].extend((surf, c) for c in cands)

for ch in list(PATTERNS_BY_FIRST.keys()):
    PATTERNS_BY_FIRST[ch].sort(key=lambda x: len(x[0]), reverse=True)


# ----------------------------
# Pinyin orthography helpers
# ----------------------------
def _is_front_final(fin: str) -> bool:
    return (
        fin.startswith("i")
        or fin.startswith("ü")
        or fin
        in {
            "i",
            "in",
            "ing",
            "ie",
            "ia",
            "ian",
            "iang",
            "iao",
            "iu",
            "iong",
            "ü",
            "üe",
            "üan",
            "ün",
        }
    )


def _spell_pinyin(initial: str, final: str) -> str:
    if initial in {"j", "q", "x"} and final.startswith("ü"):
        final = "u" + final[1:]

    if final == "iou":
        final = "iu"
    elif final == "uei":
        final = "ui"
    elif final == "uen":
        final = "un"

    if not initial:
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
        if final == "o":
            return "wo"
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

        return final

    return initial + final


def _choose_initial(ini_surf: str, fin_pin: str, require_retro: Optional[bool]) -> str:
    cands = INI_T2P_CAND.get(ini_surf, [""])
    if len(cands) == 1:
        return cands[0]

    if ini_surf == "pr":
        return "zh" if require_retro is True else "z"
    if ini_surf == "tr":
        return "ch" if require_retro is True else "c"
    if ini_surf == "cr":
        return "sh" if require_retro is True else "s"

    front = _is_front_final(fin_pin)
    if ini_surf == "g":
        return "j" if front else "g"
    if ini_surf == "c":
        return "q" if front else "k"
    if ini_surf == "ch":
        return "x" if front else "h"
    return cands[0]


def _valid_combo(initial: str, final: str) -> bool:
    # --- disambiguate o vs uo by pinyin legality ---
    if final == "o":
        return initial in {"b", "p", "m", "f"}  # forbid zero-initial 'o'
    if final == "uo":
        if initial in {"b", "p", "m", "f"}:
            return False

    retro = initial in {"zh", "ch", "sh", "r"}
    dental = initial in {"z", "c", "s"}

    if (retro or dental) and final.startswith("i") and final != "i":
        return False
    if (retro or dental) and final.startswith("ü"):
        return False
    if retro and final in {"ie", "iu", "iong", "iao", "ian", "iang"}:
        return False
    return True


def _cand_has_legal_pinyin(c: Cand) -> bool:
    ini_choices = INI_T2P_CAND.get(c.ini_surf, [""])
    if c.require_retro is True:
        if c.ini_surf == "pr":
            ini_choices = ["zh"]
        elif c.ini_surf == "tr":
            ini_choices = ["ch"]
        elif c.ini_surf == "cr":
            ini_choices = ["sh"]
        elif c.ini_surf == "r":
            ini_choices = ["r"]
        else:
            return False

    for ini in ini_choices:
        if _valid_combo(ini, c.fin_pin):
            return True
    return False


# ----------------------------
# DP decode one token word
# ----------------------------
def latin_word_to_pinyin(word: str) -> Optional[str]:
    if not word:
        return None
    w = word.lower()

    if not re.search(r"[a-zéêúèà]", w):
        return None

    n = len(w)
    INF = 10**9
    dp = [INF] * (n + 1)
    prev: List[Optional[Tuple[int, Cand]]] = [None] * (n + 1)
    dp[0] = 0

    for i in range(n):
        if dp[i] >= INF:
            continue
        ch = w[i]
        for surf, cand in PATTERNS_BY_FIRST.get(ch, []):
            j = i + len(surf)
            if j > n:
                continue
            if not w.startswith(surf, i):
                continue

            if not _cand_has_legal_pinyin(cand):
                continue

            cost = dp[i] + 1.0
            if not cand.explicit_tone:
                cost += 0.2
            if i == 0 and cand.ini_surf == "":
                cost += 0.35

            is_end = j == n
            next_vowel = _looks_vowel_start(w, j)

            if cand.tone == 2:
                if cand.marker == "st" and not is_end:
                    cost += 1.2
                if cand.marker == "s":
                    if is_end:
                        cost += 1.2
                    if not next_vowel:
                        cost += 0.8
                if cand.marker == "so":
                    if is_end:
                        cost += 0.8
                    if next_vowel:
                        cost += 0.5

            if cand.tone == 3:
                if cand.marker == "ss":
                    if is_end:
                        cost += 1.0
                    if not next_vowel:
                        cost += 0.7
                if cand.marker in {"t", "ggo", "nno"} and not is_end:
                    cost += 1.0

            if cand.tone == 4:
                if cand.marker == "v":
                    if is_end:
                        cost += 0.9
                    if not next_vowel:
                        cost += 0.7
                if cand.marker == "vo":
                    if (not is_end) and next_vowel:
                        cost += 0.4

            if cand.tone == 5 and not is_end:
                cost += 0.05
            if cand.geminated_ini:
                cost += 0.05

            if cost < dp[j]:
                dp[j] = cost
                prev[j] = (i, cand)

    if dp[n] >= INF or prev[n] is None:
        return None

    # reconstruct (IMPORTANT: copy cands so we never mutate global cache objects)
    cands: List[Cand] = []
    cur = n
    while cur > 0:
        pi, c = prev[cur]
        cands.append(c)
        cur = pi
    cands.reverse()

    # make them independent objects
    cands = [replace(c) for c in cands]

    # infer 3rd tone before consonant via gemination on CURRENT syllable
    # (NEVER mutate shared/global Cand objects)
    for k in range(1, len(cands)):
        if cands[k].geminated_ini and (not cands[k - 1].explicit_tone):
            cands[k - 1] = replace(cands[k - 1], tone=3)

    out = []
    for c in cands:
        ini_pin = _choose_initial(c.ini_surf, c.fin_pin, c.require_retro)

        if c.require_retro is True and c.ini_surf in {"pr", "tr", "cr"}:
            ini_pin = {"pr": "zh", "tr": "ch", "cr": "sh"}[c.ini_surf]

        if (
            not _valid_combo(ini_pin, c.fin_pin)
            and c.ini_surf in {"pr", "tr", "cr"}
            and c.require_retro is None
        ):
            alt = {"pr": "zh", "tr": "ch", "cr": "sh"}[c.ini_surf]
            if _valid_combo(alt, c.fin_pin):
                ini_pin = alt

        if not _valid_combo(ini_pin, c.fin_pin):
            return None

        syll = _spell_pinyin(ini_pin, c.fin_pin)
        out.append(f"{syll}{c.tone}")

    return "-".join(out)


# ----------------------------
# Convert full text: keep whitespace/punct; convert scheme-words
# ----------------------------
TOKEN_RE = re.compile(r"(\s+|[^\s]+)")
SCHEME_LETTERS_RE = r"A-Za-zéêúèàÉÊÚÈÀ"


def latin_text_to_pinyin(text: str) -> str:
    out = []
    for tok in TOKEN_RE.findall(text):
        if tok.isspace():
            out.append(tok)
            continue

        m = re.fullmatch(
            rf"([^0-9{SCHEME_LETTERS_RE}]+)?"
            rf"([{SCHEME_LETTERS_RE}]+)"
            rf"([^0-9{SCHEME_LETTERS_RE}]+)?",
            tok,
        )
        if not m:
            out.append(tok)
            continue

        pre, core, suf = m.group(1) or "", m.group(2), m.group(3) or ""
        converted = latin_word_to_pinyin(core)
        out.append(pre + (converted if converted is not None else core) + suf)

    return "".join(out)


if __name__ == "__main__":
    import sys

    s = sys.stdin.read()
    print(latin_text_to_pinyin(s))
