# poutto_deva.py
import re
from typing import Dict, Tuple, Optional, List

# ----------------------------
# Tone-mark parsing
# ----------------------------
_TONE_MARK_MAP = {
    "ā": ("a", 1),
    "á": ("a", 2),
    "ǎ": ("a", 3),
    "à": ("a", 4),
    "ē": ("e", 1),
    "é": ("e", 2),
    "ě": ("e", 3),
    "è": ("e", 4),
    "ī": ("i", 1),
    "í": ("i", 2),
    "ǐ": ("i", 3),
    "ì": ("i", 4),
    "ō": ("o", 1),
    "ó": ("o", 2),
    "ǒ": ("o", 3),
    "ò": ("o", 4),
    "ū": ("u", 1),
    "ú": ("u", 2),
    "ǔ": ("u", 3),
    "ù": ("u", 4),
    "ǖ": ("ü", 1),
    "ǘ": ("ü", 2),
    "ǚ": ("ü", 3),
    "ǜ": ("ü", 4),
    "ń": ("n", 2),
    "ň": ("n", 3),
    "ǹ": ("n", 4),
    "ḿ": ("m", 2),
}


def _strip_tone_marks(s: str) -> Tuple[str, Optional[int]]:
    tone = None
    out = []
    for ch in s:
        if ch in _TONE_MARK_MAP:
            base, t = _TONE_MARK_MAP[ch]
            out.append(base)
            tone = t if tone is None else tone
        else:
            out.append(ch)
    return "".join(out), tone


def _parse_syllable(raw: str) -> Tuple[str, int]:
    s = raw.strip().lower().replace("u:", "ü").replace("v", "ü")
    m = re.fullmatch(r"([a-zü']+)([0-5])?", s)
    if m:
        base = m.group(1)
        tone = int(m.group(2)) if m.group(2) else None
        base2, t2 = _strip_tone_marks(base)
        base = base2
        if tone is None:
            tone = t2 if t2 is not None else 1
        return base, tone
    s2, t2 = _strip_tone_marks(s)
    s2 = re.sub(r"[^a-zü]", "", s2)
    return s2, (t2 if t2 is not None else 1)


# ----------------------------
# Pinyin -> (initial, final)
# ----------------------------
_PINYIN_INITIALS = [
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
]


def _split_initial_final(p: str) -> Tuple[str, str]:
    if p.startswith("y"):
        r = p[1:]
        ymap = {
            "i": "i",
            "in": "in",
            "ing": "ing",
            "a": "ia",
            "an": "ian",
            "ang": "iang",
            "ao": "iao",
            "e": "ie",
            "ong": "iong",
            "ou": "iou",
            "o": "io",  # yo -> io
            "u": "ü",
            "ue": "üe",
            "uan": "üan",
            "un": "ün",
        }
        if r in ymap:
            return "", ymap[r]

    if p.startswith("w"):
        r = p[1:]
        wmap = {
            "u": "u",
            "a": "ua",
            "ai": "uai",
            "an": "uan",
            "ang": "uang",
            "o": "uo",
            "ei": "ui",
            "en": "un",
            "eng": "ueng",
        }
        if r in wmap:
            return "", wmap[r]

    for ini in _PINYIN_INITIALS:
        if p.startswith(ini):
            return ini, p[len(ini) :]
    return "", p


def _normalize_umlaut(ini: str, fin: str) -> str:
    if ini in {"j", "q", "x"} and fin in {"u", "ue", "uan", "un"}:
        return "ü" + fin[1:]
    return fin


# ----------------------------
# Devanagari mapping (from your document)
# Initials as BASE consonants (no virama here)
# ----------------------------
VIR = "्"

_INIT_BASE: Dict[str, str] = {
    "m": "म",
    "n": "न",
    "b": "प",
    "p": "फ",
    "d": "त",
    "t": "थ",
    "g": "क",
    "k": "ख",
    "f": "फ़",
    "s": "स",
    "x": "श",
    "sh": "ष",
    "h": "ह",
    "z": "च",
    "c": "छ",
    "j": "च",
    "q": "छ",
    "zh": "ट",
    "ch": "ठ",
    "l": "ल",
    "r": "ञ",
    "": "र",  # zero initial carrier consonant
}

# Voiced mapping on BASE consonants
_VOICED_BASE = {
    "प": "ब",
    "फ": "भ",
    "त": "द",
    "थ": "ध",
    "क": "ग",
    "ख": "घ",
    "च": "ज",
    "छ": "झ",
    "ट": "ड",
    "ठ": "ढ",
}

# Finals are given in independent shapes; if two forms -> second is word-final.
_FIN_DEV: Dict[str, Tuple[str, str]] = {
    # single
    "i": ("ई", "ई"),
    "ü": ("ॠ", "ॠ"),
    "u": ("ऊ", "ऊ"),
    "e": ("अ", "अ"),
    "o": ("ओ", "ओ"),
    "a": ("आ", "आ"),
    # apical handled separately
    # diphthongs
    "ie": ("ये", "ये"),
    "iou": ("यु", "यु"),
    "iu": ("यु", "यु"),  # harmless; will never match, but kept
    "ia": ("या", "या"),
    "iao": ("यौ", "यौ"),
    "io": ("यौ", "यौ"),
    "üe": ("ऋ", "ऋ"),
    "ue": ("ऋ", "ऋ"),
    "ui": ("वे", "वे"),
    "uo": ("ओ", "ओ"),
    "ua": ("वा", "वा"),
    "uai": ("वै", "वै"),
    "ou": ("उ", "उ"),
    "ei": ("ए", "ए"),
    "ai": ("ऐ", "ऐ"),
    "ao": ("औ", "औ"),
    # nasal
    "in": ("ईन्", "ईं"),
    "ing": ("ईम्", "ईण"),
    "ian": ("यान्", "यां"),
    "iang": ("याम्", "याण"),
    "ün": ("ॠं", "ॠं"),
    "üan": ("ऋं", "ऋं"),
    "iong": ("यों", "यों"),
    "uen": ("वन्", "वं"),
    "un": ("वन्", "वं"),
    "ueng": ("वम्", "वण"),
    "uan": ("वान्", "वां"),
    "uang": ("वाम्", "वाण"),
    "ong": ("ओं", "ओं"),
    "en": ("अन्", "अं"),
    "eng": ("अम्", "अण"),
    "an": ("आन्", "आं"),
    "ang": ("आम्", "आण"),
    # r finals
    "er": ("अञ्", "अञ्"),
    "r": ("ञ्", "ञ्"),
}

# vowel letters -> matras
VOWEL_TO_MATRA = {
    "अ": "",
    "आ": "ा",
    "इ": "ि",
    "ई": "ी",
    "उ": "ु",
    "ऊ": "ू",
    "ए": "े",
    "ऐ": "ै",
    "ओ": "ो",
    "औ": "ौ",
    "ऋ": "ृ",
    "ॠ": "ॄ",
}

VOWEL_LETTERS = set(VOWEL_TO_MATRA.keys())


def _map_final(ini: str, fin: str, is_word_final: bool) -> str:
    # apical i -> इ
    if fin == "i" and ini in {"z", "c", "s", "zh", "ch", "sh", "r"}:
        return "इ"
    if fin == "weng":
        fin = "ueng"
    if fin not in _FIN_DEV:
        raise ValueError(f"Unknown final: ini={ini}, fin={fin}")
    mid, end = _FIN_DEV[fin]
    return end if is_word_final else mid


def _combine(init_str: str, fin_str: str, *, zero_initial: bool) -> str:
    """
    Combine initial string (may contain viramas) with final independent string.
    - If final starts with vowel letter: convert to matra, attach to last consonant (remove trailing virama if any)
    - If final starts with consonant: ensure init ends with virama, then concatenate -> conjunct
    - Special: zero-initial + leading ऋ/ॠ => क्ष/ज्ञ
    """
    if not fin_str:
        return init_str

    first = fin_str[0]

    # ✅ NEW: zero-initial + (य…/व…) should NOT be prefixed by र्.
    # e.g. "र् + यौ" -> "यौ", "र् + वा" -> "वा"
    if zero_initial and first in ("य", "व"):
        return fin_str

    # zero-initial special: र्+ऋ => क्ष ; र्+ॠ => ज्ञ
    if zero_initial and first in ("ऋ", "ॠ"):
        repl = "क्ष" if first == "ऋ" else "ज्ञ"
        return repl + fin_str[1:]

    if first in VOWEL_LETTERS:
        matra = VOWEL_TO_MATRA[first]
        rest = fin_str[1:]

        # make initial "alive" (drop trailing virama if present)
        if init_str.endswith(VIR):
            init_alive = init_str[:-1]
        else:
            init_alive = init_str

        return init_alive + matra + rest

    # consonant-leading finals: need conjunct (ensure init ends with virama)
    if not init_str.endswith(VIR):
        init_str = init_str + VIR
    return init_str + fin_str


def _add_r_marker(init_base: str, *, word_initial: bool) -> str:
    """
    Add र् marker when no voiced counterpart exists.
    - word-internal: र् + C् ...
    - word-initial:  C् + र् ...
    """
    if word_initial:
        # C् + र्
        return init_base + VIR + "र" + VIR
    else:
        # र् + C्
        return "र" + VIR + init_base + VIR


def _apply_tone_to_initial(
    ini: str, init_base: str, tone: int, *, word_initial: bool
) -> str:
    """
    Apply tone effect to initial only:
      - tone2/tone4: voice initial if possible else add र् (with word-initial placement rule)
      - otherwise: unchanged
    Returns an initial string (may include virama(s)).
    """
    # zero initial: don't add extra r-marker; carrier is already र
    if ini == "":
        return init_base

    if tone in (2, 4) and ini in ("m", "n"):
        return "ह" + VIR + ("म" if ini == "m" else "न")

    if tone in (2, 4):
        if init_base in _VOICED_BASE:
            return _VOICED_BASE[init_base]
        else:
            return _add_r_marker(init_base, word_initial=word_initial)

    return init_base


def convert_word_deva(word: str, syll_sep: str = "-") -> str:
    """
    Pinyin word -> Devanagari.
    Tone rules (per your doc + latest clarifications):
      - 1: none
      - 2: voice initial or add र् (word-initial: after consonant)
      - 3: add स्
      - 4: voice initial or add र् (word-initial: after consonant) + add स्
      - 0/5: add ः
    """
    parts = re.split(rf"[{re.escape(syll_sep)}']+", word)
    parts = [p for p in parts if p]
    out_sylls: List[str] = []
    ZWNJ = "\u200c"  # Zero Width Non-Joiner

    for i, raw in enumerate(parts):
        base, tone = _parse_syllable(raw)
        ini, fin = _split_initial_final(base)
        fin = _normalize_umlaut(ini, fin)

        if ini not in _INIT_BASE:
            raise ValueError(f"Unknown initial: {ini}")

        init_base = _INIT_BASE[ini]
        fin_str = _map_final(ini, fin, is_word_final=(i == len(parts) - 1))

        # apply initial changes for tone
        init_str = _apply_tone_to_initial(ini, init_base, tone, word_initial=(i == 0))

        # for zero initial, we sometimes need r-virama if final is consonant-leading
        # _combine will add virama when needed; but if initial is "र" and fin begins with consonant,
        # it will become "र्" correctly.
        syl = _combine(init_str, fin_str, zero_initial=(ini == ""))

        # apply tone suffix
        if tone in (0, 5):
            syl += "ः"
        elif tone == 3:
            syl += "क्त"  # ✅ 关键：在 क्त 前加 ZWNJ
        elif tone == 4:
            syl += "क्त"  # ✅ 同上
        # tone1 and tone2: no suffix

        out_sylls.append(syl)

    return "".join(out_sylls)


def convert_pinyin_deva(text: str, syll_sep: str = "-") -> str:
    text = text.replace("u:", "ü").replace("U:", "Ü")
    pattern = re.compile(r"(\s+|[A-Za-züÜv0-5']+(?:-[A-Za-züÜv0-5']+)*|.)")

    out = []
    for m in pattern.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            out.append(tok)
        elif re.fullmatch(r"[A-Za-züÜv0-5']+(?:-[A-Za-züÜv0-5']+)*", tok):
            out.append(convert_word_deva(tok, syll_sep=syll_sep))
        else:
            out.append(tok)
    return "".join(out)


if __name__ == "__main__":
    import sys

    print(convert_pinyin_deva(sys.stdin.read()))
