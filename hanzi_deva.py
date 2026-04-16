# deva_reverse.py
# Devanagari-scheme -> Pinyin(with tone digits), syllables joined by '-'
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# You must have poutto_deva.py in the same folder
from poutto_deva import convert_word_deva

# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------
CTRL_CHARS = {
    "\u200c",
    "\u200d",
    "\u2060",
    "\u200b",
    "\ufeff",
    "\u00ad",
    "\u00a0",
    "\u202f",
}  # ZWNJ/ZWJ/WJ + common invisible chars


def _norm(s: str) -> str:
    # NFC to stabilize matras/halant order
    s = unicodedata.normalize("NFC", s)
    # drop invisible / shaping controls and odd spaces that can appear after copy-paste
    for ch in CTRL_CHARS:
        s = s.replace(ch, "")
    return s


# Map Devanagari punctuation back to western (optional, but nicer for pinyin)
PUNC_REV = {
    "।": ".",
    "॥": ".",
}


# ------------------------------------------------------------
# Candidate representation
# ------------------------------------------------------------
@dataclass(frozen=True)
class Cand:
    pinyin_tone: str  # e.g., "qing3"
    base: str  # e.g., "qing"
    tone: int  # 1..5


# ------------------------------------------------------------
# Pinyin normalization (standard orthography) + legality pruning
# ------------------------------------------------------------
# We normalize zero-initial syllables to standard pinyin spellings:
#   i->yi, u->wu, ua->wa, iang->yang, etc.
# We also disambiguate o vs uo (both can map to the same surface in your scheme):
#   - 'o' only after b/p/m/f
#   - otherwise use 'uo'; and zero-initial -> 'wo'

_PINYIN_INI_LIST = [
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
    "y",
    "w",
]
_PINYIN_INI_LIST.sort(key=len, reverse=True)

_YMAP = {
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
    "u": "ü",
    "ue": "üe",
    "uan": "üan",
    "un": "ün",
}
_WMAP = {
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


def _split_ini_fin(base: str) -> Tuple[str, str]:
    for ini in _PINYIN_INI_LIST:
        if ini and base.startswith(ini):
            return ini, base[len(ini) :]
    return "", base


def _orth_to_underlying(ini: str, fin: str) -> Tuple[str, str]:
    if ini == "y":
        return "", _YMAP.get(fin, fin)
    if ini == "w":
        return "", _WMAP.get(fin, fin)
    return ini, fin


def _normalize_umlaut_under(ini: str, fin: str) -> str:
    # j/q/x + u/ue/uan/un are underlying ü/üe/üan/ün
    if ini in {"j", "q", "x"} and fin in {"u", "ue", "uan", "un"}:
        return "ü" + fin[1:]
    return fin


def _is_front_final(fin: str) -> bool:
    return (
        fin.startswith("i")
        or fin.startswith("ü")
        or fin
        in {"in", "ing", "ian", "iang", "iao", "ie", "iu", "iong", "ün", "üan", "üe"}
    )


def _valid_combo(ini: str, fin: str) -> bool:
    retro = ini in {"zh", "ch", "sh", "r"}
    dental = ini in {"z", "c", "s"}
    palatal = ini in {"j", "q", "x"}

    # o vs uo
    if fin == "o":
        if ini == "":
            return False
        return ini in {"b", "p", "m", "f"}
    if fin == "uo":
        if ini in {"b", "p", "m", "f"}:
            return False

    # dental/retro restrictions (pinyin legality)
    if (retro or dental) and fin.startswith("i") and fin != "i":
        return False
    if (retro or dental) and fin.startswith("ü"):
        return False

    # palatal must be front-final/ü-family (or i itself)
    if palatal:
        if fin == "i":
            return True
        if not _is_front_final(fin):
            return False

    return True


def _disambig_o_uo(ini: str, fin: str) -> str:
    # Force o only after b/p/m/f; otherwise treat as uo
    if fin == "o" and ini not in {"b", "p", "m", "f"}:
        return "uo"
    if fin == "uo" and ini in {"b", "p", "m", "f"}:
        return "o"
    if fin == "o" and ini == "":
        return "uo"
    return fin


def _spell_pinyin(ini: str, fin: str) -> str:
    # j/q/x + ü written as u in standard pinyin
    if ini in {"j", "q", "x"} and fin.startswith("ü"):
        fin = "u" + fin[1:]

    # spelling contractions (optional, but standard)
    if fin == "iou":
        fin_sp = "iu"
    elif fin == "uei":
        fin_sp = "ui"
    elif fin == "uen":
        fin_sp = "un"
    else:
        fin_sp = fin

    if ini:
        return ini + fin_sp

    # zero-initial y/w orthography
    if fin_sp == "i":
        return "yi"
    if fin_sp == "in":
        return "yin"
    if fin_sp == "ing":
        return "ying"
    if fin_sp == "ie":
        return "ye"
    if fin_sp in {"iu", "iou"}:
        return "you"
    if fin_sp == "ia":
        return "ya"
    if fin_sp == "ian":
        return "yan"
    if fin_sp == "iang":
        return "yang"
    if fin_sp == "iao":
        return "yao"
    if fin_sp == "iong":
        return "yong"
    if fin_sp.startswith("i"):
        return "y" + fin_sp

    if fin_sp == "ü":
        return "yu"
    if fin_sp == "üe":
        return "yue"
    if fin_sp == "üan":
        return "yuan"
    if fin_sp == "ün":
        return "yun"
    if fin_sp.startswith("ü"):
        return "yu" + fin_sp[1:]

    if fin_sp == "u":
        return "wu"
    if fin_sp in {"o", "uo"}:
        return "wo"
    if fin_sp == "ua":
        return "wa"
    if fin_sp == "uai":
        return "wai"
    if fin_sp == "uan":
        return "wan"
    if fin_sp == "uang":
        return "wang"
    if fin_sp == "ui":
        return "wei"
    if fin_sp == "un":
        return "wen"
    if fin_sp == "ueng":
        return "weng"
    if fin_sp.startswith("u"):
        return "w" + fin_sp

    return fin_sp


def _normalize_cand(cand: Cand) -> Optional[str]:
    ini0, fin0 = _split_ini_fin(cand.base)
    ini1, fin1 = _orth_to_underlying(ini0, fin0)
    fin2 = _normalize_umlaut_under(ini1, fin1)
    fin3 = _disambig_o_uo(ini1, fin2)

    # merged initials (your Deva scheme merges z/j and c/q in some contexts)
    if ini1 in {"z", "c"} and _is_front_final(fin3) and fin3 != "i":
        ini1 = "j" if ini1 == "z" else "q"
        fin3 = _normalize_umlaut_under(ini1, fin3)
    if ini1 in {"j", "q"} and (not _is_front_final(fin3)) and fin3 != "i":
        ini1 = "z" if ini1 == "j" else "c"

    if not _valid_combo(ini1, fin3):
        return None

    spelled = _spell_pinyin(ini1, fin3)
    return f"{spelled}{cand.tone}"


# ------------------------------------------------------------
# Build surface dictionary via forward converter (most reliable)
# ------------------------------------------------------------
DUMMY = "ma1"  # safe dummy syllable (non-zero initial; stable across contexts)


def _dummy_final() -> str:
    return _norm(convert_word_deva(DUMMY))


def _dummy_nonfinal_initial() -> str:
    # output of first syllable in "a1-a1"
    out = _norm(convert_word_deva(f"{DUMMY}-{DUMMY}"))
    suf = _dummy_final()
    return out[: -len(suf)] if suf and out.endswith(suf) else out


_DF = None
_DNF = None


def _ensure_dummy():
    global _DF, _DNF
    if _DF is None:
        _DF = _dummy_final()
    if _DNF is None:
        _DNF = _dummy_nonfinal_initial()


def render_surface(pinyin_tone: str, is_initial: bool, is_final: bool) -> str:
    """
    Extract the exact Devanagari surface for ONE syllable in a given context:
      is_initial: syllable is first in word?
      is_final:   syllable is last in word?
    This captures your word-initial vs word-medial r-marker placement and final-vs-nonfinal finals.
    """
    _ensure_dummy()
    df = _DF
    dnf = _DNF

    if is_initial and is_final:
        return _norm(convert_word_deva(pinyin_tone))

    if is_initial and (not is_final):
        out = _norm(convert_word_deva(f"{pinyin_tone}-{DUMMY}"))
        return out[: -len(df)] if df and out.endswith(df) else out

    if (not is_initial) and is_final:
        out = _norm(convert_word_deva(f"{DUMMY}-{pinyin_tone}"))
        return out[len(dnf) :] if dnf and out.startswith(dnf) else out

    # not initial, not final
    out = _norm(convert_word_deva(f"{DUMMY}-{pinyin_tone}-{DUMMY}"))
    if dnf and out.startswith(dnf):
        out = out[len(dnf) :]
    if df and out.endswith(df):
        out = out[: -len(df)]
    return out


# ------------------------------------------------------------
# Enumerate possible pinyin syllables by brute filtering
# ------------------------------------------------------------
PINYIN_INITIALS = [
    "",
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
    "zh",
    "ch",
    "sh",
    "y",
    "w",  # include y/w orthography
]

# A superset; invalid ones will be filtered by convert_word_deva() raising errors
PINYIN_FINALS = [
    "a",
    "o",
    "e",
    "ai",
    "ei",
    "ao",
    "ou",
    "an",
    "en",
    "ang",
    "eng",
    "ong",
    "er",
    "i",
    "ia",
    "ie",
    "iao",
    "iu",
    "ian",
    "in",
    "iang",
    "ing",
    "iong",
    "u",
    "ua",
    "uo",
    "uai",
    "ui",
    "uan",
    "un",
    "uang",
    "ueng",
    "ü",
    "üe",
    "üan",
    "ün",
    # sometimes appear in your code paths
    "iou",
    "io",
]

TONES = [1, 2, 3, 4, 5]  # 5 = neutral


def _try_convert(s: str) -> bool:
    try:
        convert_word_deva(s)
        return True
    except Exception:
        return False


def build_candidates() -> List[Cand]:
    cands: List[Cand] = []
    seen = set()

    for ini in PINYIN_INITIALS:
        for fin in PINYIN_FINALS:
            base = ini + fin
            # skip impossible like empty+"" etc
            if base == "":
                continue

            # quick check with tone1
            if not _try_convert(base + "1"):
                continue

            for t in TONES:
                pt = f"{base}{t}"
                if _try_convert(pt):
                    key = (base, t)
                    if key not in seen:
                        seen.add(key)
                        cands.append(Cand(pinyin_tone=pt, base=base, tone=t))
    return cands


# ------------------------------------------------------------
# Trie for matching surfaces
# Node: dict[char -> node], plus end lists for (final/nonfinal)
# ------------------------------------------------------------
class TrieNode(dict):
    __slots__ = ("end_final", "end_nonfinal")

    def __init__(self):
        super().__init__()
        self.end_final: List[Cand] = []
        self.end_nonfinal: List[Cand] = []


def _insert(trie: TrieNode, s: str, cand: Cand, is_final: bool):
    node = trie
    for ch in s:
        nxt = node.get(ch)
        if nxt is None:
            nxt = TrieNode()
            node[ch] = nxt
        node = nxt
    (node.end_final if is_final else node.end_nonfinal).append(cand)


_TRIE_INIT_TRUE: Optional[TrieNode] = None
_TRIE_INIT_FALSE: Optional[TrieNode] = None
_MAXLEN_INIT_TRUE = 0
_MAXLEN_INIT_FALSE = 0


def _build_tries():
    global _TRIE_INIT_TRUE, _TRIE_INIT_FALSE, _MAXLEN_INIT_TRUE, _MAXLEN_INIT_FALSE
    if _TRIE_INIT_TRUE is not None:
        return

    _ensure_dummy()
    cands = build_candidates()

    trie0 = TrieNode()  # word-initial syllable
    trie1 = TrieNode()  # non-initial syllable
    max0 = 0
    max1 = 0

    for cand in cands:
        # 4 contexts
        for is_initial in (True, False):
            for is_final in (True, False):
                surf = render_surface(
                    cand.pinyin_tone, is_initial=is_initial, is_final=is_final
                )
                if not surf:
                    continue
                # store
                if is_initial:
                    _insert(trie0, surf, cand, is_final=is_final)
                    max0 = max(max0, len(surf))
                else:
                    _insert(trie1, surf, cand, is_final=is_final)
                    max1 = max(max1, len(surf))

    _TRIE_INIT_TRUE = trie0
    _TRIE_INIT_FALSE = trie1
    _MAXLEN_INIT_TRUE = max0
    _MAXLEN_INIT_FALSE = max1


# ------------------------------------------------------------
# DP decode one Devanagari "word" (no spaces)
# ------------------------------------------------------------
def decode_deva_word(word: str) -> str:
    _build_tries()
    assert _TRIE_INIT_TRUE is not None and _TRIE_INIT_FALSE is not None

    w = _norm(word)
    n = len(w)
    if n == 0:
        return ""

    # dp[i] = (cost, prev_i, cand)
    INF = 10**18
    dp_cost = [INF] * (n + 1)
    dp_prev = [-1] * (n + 1)
    dp_cand: List[Optional[Cand]] = [None] * (n + 1)

    dp_cost[0] = 0

    for i in range(n):
        if dp_cost[i] >= INF:
            continue

        trie = _TRIE_INIT_TRUE if i == 0 else _TRIE_INIT_FALSE
        node = trie
        # walk forward
        for j in range(i, n):
            ch = w[j]
            node = node.get(ch)
            if node is None:
                break

            is_final = j + 1 == n
            cands = node.end_final if is_final else node.end_nonfinal
            if not cands:
                continue

            # scoring: prefer fewer syllables, and prefer longer matches
            seg_len = (j + 1) - i
            for cand in cands:
                # base penalty per syllable
                cost = dp_cost[i] + 100 - seg_len  # longer segment -> smaller cost
                # tiny preference: tone digits explicit are already in cand
                if cost < dp_cost[j + 1]:
                    dp_cost[j + 1] = cost
                    dp_prev[j + 1] = i
                    dp_cand[j + 1] = cand

    if dp_cost[n] >= INF:
        # fallback: cannot decode
        return f"[{word}]"

    # reconstruct
    out: List[str] = []
    cur = n
    while cur > 0:
        cand = dp_cand[cur]
        if cand is None:
            break
        normed = _normalize_cand(cand)
        out.append(normed if normed is not None else cand.pinyin_tone)
        cur = dp_prev[cur]
    out.reverse()
    return "-".join(out)


# ------------------------------------------------------------
# Public API: decode full text preserving whitespace/punctuation
# ------------------------------------------------------------
DEV_RUN_RE = re.compile(
    r"[\u0900-\u097F\u200c\u200d\u2060\u200b\ufeff\u00ad\u00a0\u202f]+"
)

# Characters that should be treated as punctuation inside Devanagari runs.
# (DO NOT include visarga "ः" because it is part of syllables in your system!)
DEV_PUNC = {"।", "॥", "॰"}


def _decode_deva_run(run: str) -> str:
    """
    Decode a continuous Devanagari run that may contain both words and punctuation
    (e.g., 'दौक्त।' or 'तोंशी॥').
    We split on DEV_PUNC, decode each word chunk, and keep punctuation (mapped via PUNC_REV).
    """
    buf: List[str] = []
    out_parts: List[str] = []
    for ch in run:
        if ch in DEV_PUNC:
            if buf:
                out_parts.append(decode_deva_word("".join(buf)))
                buf.clear()
            out_parts.append(PUNC_REV.get(ch, ch))
        else:
            buf.append(ch)
    if buf:
        out_parts.append(decode_deva_word("".join(buf)))
    return "".join(out_parts)


def deva_text_to_pinyin(text: str) -> str:
    """
    Convert Devanagari-scheme text back to pinyin (tone digits).
    Preserves whitespace and non-Devanagari punctuation.
    Correctly handles danda/double-danda attached to the last word (no spaces), e.g. '...दौक्त।'
    """
    parts: List[str] = []
    idx = 0

    for m in DEV_RUN_RE.finditer(text):
        a, b = m.start(), m.end()
        if a > idx:
            parts.append(text[idx:a])

        run = text[a:b]
        if run and all(ch in DEV_PUNC for ch in run):
            parts.append("".join(PUNC_REV.get(ch, ch) for ch in run))
        else:
            parts.append(_decode_deva_run(run))

        idx = b

    if idx < len(text):
        parts.append(text[idx:])

    return "".join(parts)
