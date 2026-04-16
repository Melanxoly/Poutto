# poutto_greek.py
import re
from typing import Dict, Tuple, Optional
import unicodedata

SMOOTH = "\u0313"  # ̓
ROUGH = "\u0314"  # ̔

GREEK_VOWELS = set("αεηιουωΑΕΗΙΟΥΩ")  # 你方案里用到哪些元音就放哪些
BREATHING_ON_SECOND = {"αι", "ει", "οι", "υι", "αυ", "ευ", "ου"}  # lowercase


def _add_breathing_to_initial_vowel(word: str, mark: str) -> str:
    """
    Put breathing mark after the first character IF that character is a vowel.
    (Works for digraphs like ιε/υεο: breathing goes on the first vowel letter.)
    """
    if not word:
        return word

    # We check first two letters (case-insensitive) for the special digraphs.
    w2 = word[:2]
    w2_low = w2.lower()

    # only apply if the first char is a vowel (word-initial vowel situation)
    if word[0] not in GREEK_VOWELS:
        return word

    if len(word) >= 2 and w2_low in BREATHING_ON_SECOND:
        # insert after second character
        return word[0] + word[1] + mark + word[2:]
    else:
        # insert after first character
        return word[0] + mark + word[1:]


# -------- tone mark parsing (same as poutto.py) --------
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
    # y/w orthography -> zero-initial
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
            "o": "io",  # yo -> io（推荐）
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


RETROFLEX_INITIALS = {"zh", "ch", "sh", "r"}

# -------- greek finals mapping (from your file) --------
PINYIN_FINAL_TO_GREEK = {
    "i": "ι",
    "ü": "υ",
    "u": "ου",
    "e": "ε",
    "o": "ω",
    "a": "α",
    # apical handled in _map_final
    "ie": "ῃ",
    "iou": "υο",
    "iu": "υο",
    "ia": "ᾳ",
    "iao": "ῳ",
    "io": "υο",  # 你文件里 iao->io->ιω；yo->io 也落到 ω
    "üe": "υε",
    "ue": "υε",
    "ui": "οι",
    "uo": "ω",
    "ua": "οα",
    "uai": "η",
    "ou": "ευ",
    "ei": "ει",
    "ai": "αι",
    "ao": "αυ",
    # nasal
    "in": "ινο",
    "ing": "ιγο",  # 注意：ing->igo->ιγο
    "ian": "ιανο",
    "iang": "ιαγο",
    "ün": "υνο",
    "üan": "υενο",
    "iong": "υγο",
    "un": "ουνο",
    "ueng": "ουγο",
    "uan": "οανο",
    "uang": "οαγο",
    "ong": "ογο",
    "en": "ηνο",
    "eng": "ηγο",
    "an": "ανο",
    "ang": "αγο",
    # r final
    "er": "ρο",
}


# 翘舌(zh/ch/sh/r)专用韵母：pinyin final -> greek
# 这些正是你说的 false diphthongs / false nasal（与 h 无关）
PINYIN_FINAL_TO_GREEK_RETRO = {
    # false diphthongs
    "u": "ύ",
    "e": "έ",
    "o": "ῶ",
    "a": "ά",
    "ui": "οῖ",
    "uo": "ῶ",
    "ua": "οά",
    "uai": "ῆ",
    "ou": "εῦ",
    "ei": "εῖ",
    "ai": "αῖ",
    "ao": "αῦ",
    # false nasal
    "un": "οῦνο",
    "ueng": "οῦγο",
    "uan": "οάνο",
    "uang": "οάγο",
    "ong": "όγο",
    "en": "ῆνο",
    "eng": "ῆγο",
    "an": "άνο",
    "ang": "άγο",
}


def _map_final(pin_ini: str, pin_fin: str) -> str:
    # apical i
    if pin_fin == "i":
        if pin_ini in {"z", "c", "s"}:
            return "ι"  # ï1
        if pin_ini in {"zh", "ch", "sh", "r"}:
            return "ᾳ"  # ï2
        return "ι"
    if pin_fin == "weng":
        pin_fin = "ueng"
    if pin_ini in RETROFLEX_INITIALS and pin_fin in PINYIN_FINAL_TO_GREEK_RETRO:
        return PINYIN_FINAL_TO_GREEK_RETRO[pin_fin]
    else:
        if pin_fin in PINYIN_FINAL_TO_GREEK:
            return PINYIN_FINAL_TO_GREEK[pin_fin]
    raise ValueError(f"Unknown final: ini={pin_ini}, fin={pin_fin}")


# -------- greek initials mapping (from your file) --------
def _map_initial(pin_ini: str, trans_final: str, apical_retroflex: bool) -> str:
    # zero initial: you wrote "-" in greek column; use empty string
    if pin_ini == "":
        return ""

    direct = {
        "m": "μ",
        "n": "ν",
        "b": "β",
        "p": "π",
        "d": "δ",
        "t": "τ",
        "g": "γ",
        "k": "κ",
        "f": "φ",
        "s": "θ",
        "x": "χ",
        "sh": "θ",
        "h": "χ",
        "z": "ψ",
        "c": "ζ",
        "j": "γ",
        "q": "κ",
        "zh": "ψ",
        "ch": "ζ",
        "l": "λ",
        "r": "ρ",
    }
    if pin_ini in direct:
        return direct[pin_ini]

    raise ValueError(f"Unknown initial: {pin_ini}")


# -------- tone anchor + insertion on greek strings --------
_GREEK_VOWELS = set("αεηιουω")  # plus υ treated as vowel already (y/ü)


def _tone_anchor_pos_greek(trans: str) -> int:
    # exclude nasal suffix 'νος'/'γος' and r 'ρ'?? here er ends with ρ? actually "ερ"
    cut = len(trans)
    if trans.endswith(("νος", "γος")):
        cut -= 3
    elif trans.endswith("ρ"):
        cut -= 1

    idx = -1
    for i, ch in enumerate(trans[:cut]):
        if ch in _GREEK_VOWELS or ch == "υ":
            idx = i
    return idx


def syllable_info_greek(pin: str) -> Dict:
    base, tone = _parse_syllable(pin)
    ini, fin = _split_initial_final(base)
    fin = _normalize_umlaut(ini, fin)

    apical_retroflex = fin == "i" and ini in {"zh", "ch", "sh", "r"}
    trans_fin = _map_final(ini, fin)
    trans_ini = _map_initial(ini, trans_fin, apical_retroflex=apical_retroflex)
    base_trans = trans_ini + trans_fin

    vowel_start = (ini == "") or base.startswith(("y", "w"))
    is_nasal = base_trans.endswith(("νος", "γος"))
    return {
        "raw": pin,
        "pinyin": base,
        "tone": tone,
        "vowel_start": vowel_start,
        "base_trans": base_trans,
        "is_nasal": is_nasal,
    }


def convert_word_greek(word: str, syll_sep: str = "-") -> str:
    parts = re.split(rf"[{re.escape(syll_sep)}']+", word)
    parts = [p for p in parts if p]
    sylls = [syllable_info_greek(p) for p in parts]

    # ---- word-level breathing decision ----
    # Determine the pinyin initial of the FIRST syllable (before mapping).
    # We need to re-split the first syllable's plain pinyin.
    first_base, _ = _parse_syllable(parts[0])
    first_ini, _ = _split_initial_final(first_base)
    word_initial = first_ini  # "" or "h" or others

    outs = [s["base_trans"] for s in sylls]
    dup_prefix = [0] * len(sylls)

    for i, s in enumerate(sylls):
        tone = s["tone"]
        is_last = i == len(sylls) - 1
        next_is_vowel = sylls[i + 1]["vowel_start"] if not is_last else False

        # light tone
        if tone in (0, 5):
            outs[i] = outs[i] + "ς"
            continue

        # 1st tone
        if tone == 1:
            if is_last:
                outs[i] = outs[i] + "λετο"
            else:
                # NEW: only mark 1st tone in word-medial position when followed by a vowel-start syllable
                if next_is_vowel:
                    outs[i] = outs[i] + "λον"
                # else: no marker
            continue

        # 2nd tone
        if tone == 2:
            if is_last:
                outs[i] = outs[i] + "σαι"
            else:
                outs[i] = outs[i] + ("σ" if next_is_vowel else "σο")
            continue

        # 3rd tone
        if tone == 3:
            if is_last:
                outs[i] = outs[i] + "τος"
            else:
                if next_is_vowel:
                    outs[i] = outs[i] + "σσ"
                else:
                    dup_prefix[i + 1] += 1
            continue

        # 4th tone
        if tone == 4:
            if is_last or next_is_vowel:
                outs[i] = outs[i] + "ξ"
            else:
                outs[i] = outs[i] + "ξο"
            continue

    # apply 3rd-tone doubling to next syllable (greek consonants)
    SPECIAL_GEM = {"φ": "φθ", "θ": "σθ", "χ": "χθ", "ψ": "πτ", "ζ": "στ", "ξ": "κτ"}

    for i, cnt in enumerate(dup_prefix):
        if cnt <= 0:
            continue
        s2 = outs[i]
        if not s2:
            continue
        first = s2[0]
        rep = SPECIAL_GEM.get(first)

        if rep is None:
            # normal doubling: prefix the first char cnt times
            outs[i] = first * cnt + s2
        else:
            # special: replace φφ/θθ/χχ with φθ/σθ/χθ (apply once per cnt; cnt>1 rare)
            # Apply cnt times by repeatedly rewriting, but usually cnt==1.
            for _ in range(cnt):
                s2 = rep + s2[1:]
            outs[i] = s2

    greek_word = "".join(outs)

    if word_initial == "h":
        # only remove the very first consonant letter if it is χ
        if greek_word.startswith("χ"):
            greek_word = greek_word[1:]
        greek_word = _add_breathing_to_initial_vowel(greek_word, ROUGH)

    elif word_initial == "":
        # ✅ NEW POLICY: word-initial upsilon always rough breathing
        if greek_word and greek_word[0] in ("υ", "Υ"):
            greek_word = _add_breathing_to_initial_vowel(greek_word, ROUGH)
        else:
            greek_word = _add_breathing_to_initial_vowel(greek_word, SMOOTH)

    greek_word = unicodedata.normalize("NFC", greek_word)
    return greek_word
