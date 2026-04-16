# hanzi.py
# Latin-scheme (poutto.py) -> Pinyin(with tone digits), syllables joined by '-'
import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

# ----------------------------
# Config: vowels in Latin-scheme surface (must include diacritics used in poutto.py)
# ----------------------------
VOWELS = set("aeiouüéêùèà")
ASCII_LETTERS = set("abcdefghijklmnopqrstuvwxyz")
GEMMABLE_FIRST = set(
    "mnbpdtgclrc"
)  # first letter that could be doubled by 3rd-tone gemination
# (Note: we intentionally exclude 'h' from gemination; forward rules should not geminate a vowel-start syllable)

# ----------------------------
# Inverse maps: transcription-final -> pinyin-final candidates
# Built from your current poutto.py tables
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
}  # transcription initials that may represent retroflex family

# Build: trans_final -> list of (pinyin_final, require_retro)
# require_retro:
#   - True  : only valid when pinyin initial is retroflex (zh/ch/sh/r) family
#   - None  : could be used anywhere (std table)
FinalCand = Tuple[str, Optional[bool]]
FINAL_T2P: Dict[str, List[FinalCand]] = defaultdict(list)

# std inverse
for fin_pin, fin_tr in _FINAL_MAP_STD.items():
    FINAL_T2P[fin_tr].append((fin_pin, None))

# retro inverse (requires retro)
for fin_pin, fin_tr in _FINAL_MAP_RETRO.items():
    FINAL_T2P[fin_tr].append((fin_pin, True))

# Special: retroflex apical "i" (zhi/chi/shi/ri) -> trans final 'é'
# In your poutto.py: if pin_fin=='i' and ini in {zh,ch,sh,r} => return 'é'
FINAL_T2P["é"].append(("i", True))

# De-dup candidates
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
# Transcription initials (surface) -> candidate pinyin initials
# (Ambiguity resolved by final family heuristics)
# ----------------------------
INI_T2P_CAND: Dict[str, List[str]] = {
    "": [""],  # missing h (dropped) => zero initial
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

# For parsing, we only need these transcription initials (normalized).
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
    # next syllable considered vowel-start if next char is a vowel (missing h)
    # OR next char is 'h' (explicit zero-initial marker)
    if pos >= len(word):
        return False
    ch = word[pos]
    return ch == "h" or _is_vowel_start_char(ch)


# ----------------------------
# Candidate syllable record used by DP
# ----------------------------
@dataclass(frozen=True)
class Cand:
    ini_surf: str  # normalized transcription initial ("" means missing-h zero initial)
    fin_tr: str  # transcription final
    fin_pin: str  # underlying pinyin final (before y/w orthography)
    require_retro: Optional[bool]  # True / None
    tone: int  # 1..5 (5 neutral)
    explicit_tone: bool
    geminated_ini: bool
    marker: str  # 'none','light_s','s','so','st','ss','t','v','vo','ggo','nno'


# ----------------------------
# Precompute all possible syllable SURFACE forms and their Cand lists
# (including geminated forms: first char doubled)
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
                # If this final requires retro, only allow initials that can represent retro family
                if req_retro is True and ini not in RETRO_INI_SURF:
                    continue

                base = ini + fin_tr
                is_nasal = base.endswith(("no", "go"))

                # 1) tone=1 (no marker), not explicit
                _add_surface(
                    base,
                    Cand(
                        ini_surf=ini,
                        fin_tr=fin_tr,
                        fin_pin=fin_pin,
                        require_retro=req_retro,
                        tone=1,
                        explicit_tone=False,
                        geminated_ini=False,
                        marker="none",
                    ),
                )

                # 2) tone=5 (neutral/light): append 's' at end (explicit)
                _add_surface(
                    base + "s",
                    Cand(
                        ini_surf=ini,
                        fin_tr=fin_tr,
                        fin_pin=fin_pin,
                        require_retro=req_retro,
                        tone=5,
                        explicit_tone=True,
                        geminated_ini=False,
                        marker="light_s",
                    ),
                )

                # 3) tone=2: st / s / so (explicit)
                if is_nasal:
                    # nasal word-medial: marker after no/go (s / so)
                    for mk, suf in (("s", "s"), ("so", "so")):
                        _add_surface(
                            base + suf,
                            Cand(
                                ini_surf=ini,
                                fin_tr=fin_tr,
                                fin_pin=fin_pin,
                                require_retro=req_retro,
                                tone=2,
                                explicit_tone=True,
                                geminated_ini=False,
                                marker=mk,
                            ),
                        )
                    # nasal word-final: marker before no/go, always 'so'
                    if base.endswith(("no", "go")):
                        surf = base[:-2] + "so" + base[-2:]
                        _add_surface(
                            surf,
                            Cand(
                                ini_surf=ini,
                                fin_tr=fin_tr,
                                fin_pin=fin_pin,
                                require_retro=req_retro,
                                tone=2,
                                explicit_tone=True,
                                geminated_ini=False,
                                marker="so",
                            ),
                        )
                else:
                    for mk, ins in (("st", "st"), ("s", "s"), ("so", "so")):
                        surf = _insert_after_anchor(base, ins)
                        _add_surface(
                            surf,
                            Cand(
                                ini_surf=ini,
                                fin_tr=fin_tr,
                                fin_pin=fin_pin,
                                require_retro=req_retro,
                                tone=2,
                                explicit_tone=True,
                                geminated_ini=False,
                                marker=mk,
                            ),
                        )

                # 4) tone=3: ss / t (explicit), plus word-final nasal ggo/nno (explicit)
                # ss: before vowel
                if is_nasal:
                    # nasal word-medial: ss after no/go
                    surf_ss = base + "ss"
                else:
                    surf_ss = _insert_after_anchor(base, "ss")
                _add_surface(
                    surf_ss,
                    Cand(
                        ini_surf=ini,
                        fin_tr=fin_tr,
                        fin_pin=fin_pin,
                        require_retro=req_retro,
                        tone=3,
                        explicit_tone=True,
                        geminated_ini=False,
                        marker="ss",
                    ),
                )

                # t: word-final only (non-nasal)
                if not is_nasal:
                    surf_t = _insert_after_anchor(base, "t")
                    _add_surface(
                        surf_t,
                        Cand(
                            ini_surf=ini,
                            fin_tr=fin_tr,
                            fin_pin=fin_pin,
                            require_retro=req_retro,
                            tone=3,
                            explicit_tone=True,
                            geminated_ini=False,
                            marker="t",
                        ),
                    )

                # word-final nasal: consonant-before style => ggo / nno
                if base.endswith("go"):
                    _add_surface(
                        base[:-2] + "ggo",
                        Cand(
                            ini_surf=ini,
                            fin_tr=fin_tr,
                            fin_pin=fin_pin,
                            require_retro=req_retro,
                            tone=3,
                            explicit_tone=True,
                            geminated_ini=False,
                            marker="ggo",
                        ),
                    )
                if base.endswith("no"):
                    _add_surface(
                        base[:-2] + "nno",
                        Cand(
                            ini_surf=ini,
                            fin_tr=fin_tr,
                            fin_pin=fin_pin,
                            require_retro=req_retro,
                            tone=3,
                            explicit_tone=True,
                            geminated_ini=False,
                            marker="nno",
                        ),
                    )

                # 5) tone=4: v / vo (explicit)
                if is_nasal:
                    # nasal word-medial: marker after no/go
                    _add_surface(
                        base + "v",
                        Cand(
                            ini_surf=ini,
                            fin_tr=fin_tr,
                            fin_pin=fin_pin,
                            require_retro=req_retro,
                            tone=4,
                            explicit_tone=True,
                            geminated_ini=False,
                            marker="v",
                        ),
                    )
                    _add_surface(
                        base + "vo",
                        Cand(
                            ini_surf=ini,
                            fin_tr=fin_tr,
                            fin_pin=fin_pin,
                            require_retro=req_retro,
                            tone=4,
                            explicit_tone=True,
                            geminated_ini=False,
                            marker="vo",
                        ),
                    )
                    # nasal word-final: marker before no/go, always 'vo'
                    if base.endswith(("no", "go")):
                        surf = base[:-2] + "vo" + base[-2:]
                        _add_surface(
                            surf,
                            Cand(
                                ini_surf=ini,
                                fin_tr=fin_tr,
                                fin_pin=fin_pin,
                                require_retro=req_retro,
                                tone=4,
                                explicit_tone=True,
                                geminated_ini=False,
                                marker="vo",
                            ),
                        )
                else:
                    for mk, ins in (("v", "v"), ("vo", "vo")):
                        surf = _insert_after_anchor(base, ins)
                        _add_surface(
                            surf,
                            Cand(
                                ini_surf=ini,
                                fin_tr=fin_tr,
                                fin_pin=fin_pin,
                                require_retro=req_retro,
                                tone=4,
                                explicit_tone=True,
                                geminated_ini=False,
                                marker=mk,
                            ),
                        )

    # Add geminated variants: first character doubled, used to infer previous tone=3 (before consonant)
    # We DO NOT change the candidate's tone here; we only set geminated_ini flag on THIS syllable.
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

# Index patterns by first char for speed
PATTERNS_BY_FIRST: DefaultDict[str, List[Tuple[str, Cand]]] = defaultdict(list)
for surf, cands in SURFACE2CANDS.items():
    if not surf:
        continue
    PATTERNS_BY_FIRST[surf[0]].extend((surf, c) for c in cands)

# Sort longer surfaces first to reduce branching
for ch in list(PATTERNS_BY_FIRST.keys()):
    PATTERNS_BY_FIRST[ch].sort(key=lambda x: len(x[0]), reverse=True)


# ----------------------------
# Orthography: (initial, final) -> standard pinyin spelling (no tone digit)
# (same spirit as hanzi_greek.py)
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
    # j/q/x + ü => written as u
    if initial in {"j", "q", "x"} and final.startswith("ü"):
        final = "u" + final[1:]

    if final == "iou":
        final = "iu"
    elif final == "uei":
        final = "ui"
    elif final == "uen":
        final = "un"

    if not initial:
        # i-family
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

        # ü-family
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

        # u-family
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

    # pr/tr/cr: decide retro or not
    if ini_surf == "pr":
        return "zh" if require_retro is True else "z"
    if ini_surf == "tr":
        return "ch" if require_retro is True else "c"
    if ini_surf == "cr":
        return "sh" if require_retro is True else "s"

    front = _is_front_final(fin_pin)

    if ini_surf == "g":  # g vs j
        return "j" if front else "g"
    if ini_surf == "c":  # k vs q
        return "q" if front else "k"
    if ini_surf == "ch":  # h vs x
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

    # retro/dental cannot take i-family except apical 'i'
    if (retro or dental) and final.startswith("i") and final != "i":
        return False
    # retro/dental cannot take ü-family
    if (retro or dental) and final.startswith("ü"):
        return False
    # retro cannot take iu/iong/ie etc
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
# DP parse a single token word into syllables (Cand sequence)
# ----------------------------
def latin_word_to_pinyin(word: str) -> Optional[str]:
    if not word:
        return None
    w = word.lower()

    # quick filter: must contain scheme-ish letters
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

            # prefer explicit tones slightly
            if not cand.explicit_tone:
                cost += 0.2

            # missing 'h' at word start is unlikely (forward normally keeps it)
            if i == 0 and cand.ini_surf == "":
                cost += 0.3

            # marker/context consistency heuristics
            is_end = j == n
            next_vowel = _looks_vowel_start(w, j)
            is_nasal = cand.fin_tr.endswith(("no", "go"))

            if cand.marker in {"st", "t", "ggo", "nno"} and not is_end:
                cost += 0.6

            if cand.tone == 2:
                if cand.marker == "s":
                    if not next_vowel:
                        cost += 0.4
                    if is_end:
                        cost += 0.8  # 2nd tone word-final should be 'st'
                elif cand.marker == "so":
                    if next_vowel:
                        cost += 0.2
                elif cand.marker == "st":
                    if not is_end:
                        cost += 0.5

            if cand.tone == 4:
                if cand.marker == "v":
                    if not next_vowel or is_end:
                        cost += 0.4
                elif cand.marker == "vo":
                    if next_vowel and not is_end:
                        cost += 0.2

            if cand.tone == 5 and not is_end:
                continue  # light tone only appears at word end under your segmentation

            # geminated syllable itself is rare; tiny penalty to avoid over-using
            if cand.geminated_ini:
                cost += 0.05

            if cost < dp[j]:
                dp[j] = cost
                prev[j] = (i, cand)

    if dp[n] >= INF or prev[n] is None:
        return None

    # reconstruct cand list
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
    for k in range(1, len(cands)):
        if cands[k].geminated_ini and (not cands[k - 1].explicit_tone):
            cands[k - 1] = replace(cands[k - 1], tone=3)

    # emit pinyin syllables
    out = []
    for c in cands:
        ini_pin = _choose_initial(c.ini_surf, c.fin_pin, c.require_retro)

        # If require_retro True, but we somehow chose non-retro, force it for pr/tr/cr
        if c.require_retro is True and c.ini_surf in {"pr", "tr", "cr"}:
            ini_pin = {"pr": "zh", "tr": "ch", "cr": "sh"}[c.ini_surf]

        # phonotactic sanity: if invalid and ini_surf is pr/tr/cr and require_retro is None,
        # try the retro option
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
# allowed core letters for the Latin scheme
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
