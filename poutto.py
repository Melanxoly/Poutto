import re
from typing import Dict, Tuple, Optional, List

# ----------------------------
# Tone-mark parsing (optional)
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

_VOWELS = set("aeiouéêùèà")


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
    """
    Return (plain_pinyin_with_ü, tone_number).

    Accepts:
      - tone digits: niu2, qing3, de5 (or 0 as neutral)
      - tone marks: nǐ, qǐng
      - ü written as ü / v / u:
    If tone missing -> default 1 (阴平不标)
    """
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
    """
    Handle y/w as orthographic, turning them into zero-initial with rewritten finals.
    """
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
            "o": "iou",  # yo → you
            "ong": "iong",
            "ou": "iou",
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
    # j/q/x: u/ue/uan/un are actually ü/üe/üan/ün
    if ini in {"j", "q", "x"} and fin in {"u", "ue", "uan", "un"}:
        return "ü" + fin[1:]
    return fin


RETROFLEX_INITIALS = {"zh", "ch", "sh", "r"}

_FINAL_MAP_STD = {
    # single
    "i": "i",
    "ü": "u",
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
    # false diphthongs (only for retroflex initials)
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
    # false nasal (only for retroflex initials)
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


def _map_final(pin_ini: str, pin_fin: str) -> str:
    # apical finals:
    # zi/ci/si -> ï1 -> i
    # zhi/chi/shi/ri -> ï2 -> ie
    if pin_fin == "i":
        if pin_ini in {"z", "c", "s"}:
            return "i"
        if pin_ini in {"zh", "ch", "sh", "r"}:
            return "é"
        return "i"

    if pin_fin == "weng":
        pin_fin = "ueng"

    if pin_ini in RETROFLEX_INITIALS and pin_fin in _FINAL_MAP_RETRO:
        return _FINAL_MAP_RETRO[pin_fin]

    if pin_fin in _FINAL_MAP_STD:
        return _FINAL_MAP_STD[pin_fin]

    raise ValueError(f"Unknown final: ini={pin_ini}, fin={pin_fin}")


# ----------------------------
# Initials mapping + variants
# ----------------------------
def _map_initial(pin_ini: str, trans_final: str, apical_retroflex: bool) -> str:
    # zero initial
    if pin_ini == "":
        return "h"

    direct = {
        "m": "m",
        "n": "n",
        "b": "b",
        "p": "p",
        "d": "d",
        "t": "t",
        "g": "g",
        "k": "c",
        "f": "ph",
        "s": "cr",
        "x": "ch",
        "sh": "cr",
        "h": "ch",
        "z": "pr",
        "c": "tr",
        "j": "g",
        "q": "c",
        "zh": "pr",
        "ch": "tr",
        "l": "l",
        "r": "r",
    }
    if pin_ini in direct:
        return direct[pin_ini]

    raise ValueError(f"Unknown initial: {pin_ini}")


# ----------------------------
# Tone insertion helpers
# ----------------------------
def _tone_anchor_pos(trans: str) -> int:
    """
    Tone marks go after the last vowel of the NUCLEUS,
    but nasal codas are encoded as suffix 'no'/'go' where the trailing 'o'
    must NOT be treated as vowel.
    """
    cut = len(trans)
    if trans.endswith(("no", "go")):
        cut -= 2
    elif trans.endswith("r"):
        cut -= 1

    idx = -1
    for i, ch in enumerate(trans[:cut]):
        if ch in _VOWELS:
            idx = i
    return idx


def _insert_after_anchor(trans: str, ins: str) -> str:
    idx = _tone_anchor_pos(trans)
    if idx < 0:
        return trans + ins
    return trans[: idx + 1] + ins + trans[idx + 1 :]


# ----------------------------
# Core conversion
# ----------------------------
def syllable_info(pin: str) -> Dict:
    """
    Exported helper:
    returns tone, base_trans, vowel_start, is_nasal (based on no/go)
    """
    base, tone = _parse_syllable(pin)
    ini, fin = _split_initial_final(base)
    fin = _normalize_umlaut(ini, fin)

    apical_retroflex = fin == "i" and ini in {"zh", "ch", "sh", "r"}
    trans_fin = _map_final(ini, fin)
    trans_ini = _map_initial(ini, trans_fin, apical_retroflex=apical_retroflex)
    base_trans = trans_ini + trans_fin

    vowel_start = (ini == "") or base.startswith(("y", "w"))
    is_nasal = base_trans.endswith(("no", "go"))
    return {
        "raw": pin,
        "pinyin": base,
        "tone": tone,
        "vowel_start": vowel_start,
        "base_trans": base_trans,
        "is_nasal": is_nasal,
    }


def convert_word(word: str, syll_sep: str = "-") -> str:
    """
    One 'word' may contain multiple syllables, split by '-' or apostrophe.
    """
    parts = re.split(rf"[{re.escape(syll_sep)}']+", word)
    parts = [p for p in parts if p]
    sylls = [syllable_info(p) for p in parts]

    outs = [s["base_trans"] for s in sylls]
    dup_prefix = [0] * len(sylls)

    for i, s in enumerate(sylls):
        tone = s["tone"]
        is_last = i == len(sylls) - 1
        next_is_vowel = sylls[i + 1]["vowel_start"] if not is_last else False
        is_nasal = s.get("is_nasal", False)

        # 轻声：末尾 +s（轻声只会出现在词末；鼻尾 no/go 的 s 也自然在 no/go 之后）
        if tone in (0, 5):
            outs[i] = outs[i] + "s"
            continue

        # 二声：
        #   - 非鼻尾：维持原规则（词末 st；词中后接元音 s；后接辅音 so）
        #   - 鼻尾 no/go：
        #       * 词中：标记放在 no/go 之后（后接元音 s；后接辅音 so）
        #       * 词末：标记仍在 no/go 之前，但永远采用“词中+辅音前”形式 so
        if tone == 2:
            if is_nasal and outs[i].endswith(("no", "go")):
                if is_last:
                    outs[i] = outs[i][:-2] + "so" + outs[i][-2:]
                else:
                    outs[i] = outs[i] + ("s" if next_is_vowel else "so")
            else:
                if is_last:
                    # 词末：past
                    outs[i] = _insert_after_anchor(outs[i], "st")
                else:
                    if next_is_vowel:
                        # 词中 + 后接元音：pas
                        outs[i] = _insert_after_anchor(outs[i], "s")
                    else:
                        # 词中 + 后接辅音：paso
                        outs[i] = _insert_after_anchor(outs[i], "so")
            continue

        # 三声：
        #   - 非鼻尾：维持原规则（词末 t；词中后接元音 ss；否则倍写下一辅音）
        #   - 鼻尾 no/go：
        #       * 词中且后接元音：ss 放在 no/go 之后
        #       * 词末：标记仍在 no/go 之前，采用“词中+辅音前”形式（倍写 no/go 的辅音）=> nno/ggo
        if tone == 3:
            if is_last:
                if is_nasal:
                    if outs[i].endswith("go"):
                        outs[i] = outs[i][:-2] + "ggo"
                    elif outs[i].endswith("no"):
                        outs[i] = outs[i][:-2] + "nno"
                    else:
                        outs[i] = _insert_after_anchor(outs[i], "t")
                else:
                    outs[i] = _insert_after_anchor(outs[i], "t")
            else:
                if next_is_vowel:
                    if is_nasal:
                        outs[i] = outs[i] + "ss"
                    else:
                        outs[i] = _insert_after_anchor(outs[i], "ss")
                else:
                    dup_prefix[i + 1] += 1
            continue

        # 四声：
        #   - 非鼻尾：维持原规则（词中后接元音 v；否则 vo）
        #   - 鼻尾 no/go：
        #       * 词中：标记放在 no/go 之后（后接元音 v；后接辅音 vo）
        #       * 词末：标记仍在 no/go 之前，但永远采用“词中+辅音前”形式 vo
        if tone == 4:
            if is_nasal and outs[i].endswith(("no", "go")):
                if is_last:
                    outs[i] = outs[i][:-2] + "vo" + outs[i][-2:]
                else:
                    outs[i] = outs[i] + ("v" if next_is_vowel else "vo")
            else:
                outs[i] = _insert_after_anchor(
                    outs[i], "v" if (not is_last and next_is_vowel) else "vo"
                )
            continue

        # 一声：不标

    # apply 3rd-tone doubling to next syllable
    for i, cnt in enumerate(dup_prefix):
        if cnt > 0 and outs[i]:
            outs[i] = outs[i][0] * cnt + outs[i]

    # ----------------------------
    # Drop 'h' for zero-initial syllables
    # when NOT word-initial and previous tone is 2/3/4
    # ----------------------------
    for i in range(1, len(outs)):
        cur = outs[i]
        prev = sylls[i - 1]

        if (
            cur.startswith("h")  # 当前是零声母
            and sylls[i]["pinyin"].startswith(("a", "e", "i", "o", "u", "ü", "y", "w"))
            and prev["tone"] in (2, 3, 4)  # 前字是 2/3/4 声
        ):
            outs[i] = cur[1:]  # 去掉 h

    return "".join(outs)


def convert_pinyin(text: str, syll_sep: str = "-") -> str:
    """
    Convert a full pinyin string while preserving whitespace and punctuation.
    Recommended format:
      - spaces separate words
      - '-' separate syllables inside a word
    """
    text = text.replace("u:", "ü").replace("U:", "Ü")

    pattern = re.compile(r"(\s+|[A-Za-züÜv0-5']+(?:-[A-Za-züÜv0-5']+)*|.)")
    out = []
    for m in pattern.finditer(text):
        tok = m.group(0)
        if tok.isspace():
            out.append(tok)
        elif re.fullmatch(r"[A-Za-züÜv0-5']+(?:-[A-Za-züÜv0-5']+)*", tok):
            out.append(convert_word(tok, syll_sep=syll_sep))
        else:
            out.append(tok)
    return "".join(out)


if __name__ == "__main__":
    import sys

    text = sys.stdin.read()
    print(convert_pinyin(text))
