# hanzi_suggest.py
# ------------------------------------------------------------
# Hanzi suggestion from recovered pinyin (STRICT tone + cross-word context)
# ------------------------------------------------------------
# Forced behavior (no UI switches):
#   - Strict base+tone matching per syllable (digits 1-5), with a tiny erhua tolerance for "er" (2/5/0).
#   - Use cross-word context:
#       1) decode the whole sentence-run when feasible (strongest context)
#       2) otherwise overlapping-window voting across word boundaries
#
# Key improvements vs previous versions:
#   1) Prevent "weird but technically matchable" readings:
#        - use heteronym=True BUT penalize by reading rank (0=most common),
#          and reject matches that require too-rare pronunciations (MAX_PRON_RANK).
#      This fixes cases like guang4 -> 光 if 光 only matches via a rare reading.
#   2) Make strict match robust for polyphonic chars while still preferring common readings.
#   3) Run decoding ignores commas/quotes/colons etc; only hard sentence breaks split runs.
#
# Dependencies:
#   pip install Pinyin2Hanzi pypinyin
# Optional:
#   pip install jieba   (for a light lexicon penalty; improves semantic naturalness without hard-coding examples)

from __future__ import annotations
import re
import math
import functools
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

from pypinyin import pinyin as _pypinyin, Style

try:
    from Pinyin2Hanzi import DefaultDagParams, dag
except Exception:
    try:
        from pinyin2hanzi import DefaultDagParams, dag  # type: ignore
    except Exception:
        DefaultDagParams = None  # type: ignore
        dag = None  # type: ignore

# optional lexicon signal
try:
    import jieba  # type: ignore

    _JIEBA_FREQ = getattr(getattr(jieba, "dt", None), "FREQ", None)
except Exception:
    _JIEBA_FREQ = None


# ----------------------------
# Forced config
# ----------------------------
WINDOW_WORDS = 7
FULL_RUN_MAX_WORDS = 80

TOPK_WINDOW_BASE = 260  # base beam for windows
TOPK_FULL_BASE = 600  # base beam for full-run
TOPK_FULL_PER_WORD = 140  # additional beam per word (capped)

TOPK_CAP = 6000  # hard cap to avoid runaway

# Pronunciation rank control:
# 0 = first (most common) reading; 1 = second reading, etc.
# Reject matches that require reading rank > MAX_PRON_RANK.
MAX_PRON_RANK = 1  # allow top2 readings only (very effective for "光"这类)
LEX_W = 0.35  # lexicon penalty weight (0 disables)


# ----------------------------
# Tokenization
# ----------------------------
_PINYIN_WORD_RE = r"[A-Za-züv:]+[1-5](?:-[A-Za-züv:]+[1-5])*"
_TOKEN_RE = re.compile(rf"(\s+|{_PINYIN_WORD_RE}|.)", re.I)
_SYL_RE = re.compile(r"^([a-züv:]+)([1-5])$", re.I)
_HARD_BREAK = set(list(".!?。！？\n\r"))


def _norm_base(s: str) -> str:
    return s.lower().replace("u:", "v").replace("ü", "v")


def _parse_word(word: str) -> Optional[Tuple[List[str], List[int]]]:
    bases: List[str] = []
    tones: List[int] = []
    for syl in word.split("-"):
        syl = syl.strip()
        if not syl:
            continue
        m = _SYL_RE.match(syl)
        if not m:
            return None
        bases.append(_norm_base(m.group(1)))
        tones.append(int(m.group(2)))
    return bases, tones


def _is_word(tok: str) -> bool:
    return bool(re.fullmatch(_PINYIN_WORD_RE, tok, re.I))


def _is_hard_break(tok: str) -> bool:
    if tok.isspace():
        return ("\n" in tok) or ("\r" in tok)
    return tok in _HARD_BREAK


# ----------------------------
# Pronunciation ranked options from pypinyin
# ----------------------------
def _pypinyin_bt_per_char_ranked(hz: str) -> List[List[Tuple[str, int, int]]]:
    """
    Per character: ranked list of (base,tone,rank) from pypinyin heteronyms.
    rank=0 is the first (most common) pinyin returned by pypinyin for that char.
    """
    pys = _pypinyin(
        hz,
        style=Style.TONE3,
        neutral_tone_with_five=True,
        strict=False,
        heteronym=True,
    )
    out: List[List[Tuple[str, int, int]]] = []
    for variants in pys:
        opts: List[Tuple[str, int, int]] = []
        for rank, s in enumerate(variants):
            s = _norm_base(s)
            m = re.fullmatch(r"([a-zv]+)([0-5])", s)
            if not m:
                continue
            opts.append((m.group(1), int(m.group(2)), rank))
        if not opts:
            return []
        out.append(opts)
    return out


def _strict_match_penalty(
    bases: List[str], tones: List[int], hz: str, restrict_tone: bool = True
) -> Optional[float]:
    """
    If hz matches bases (+ tones when restrict_tone=True), return a penalty (lower better).
    Penalty is sum of matched reading ranks.
    Reject if match requires reading rank > MAX_PRON_RANK anywhere.
    """
    bt_ranked = _pypinyin_bt_per_char_ranked(hz)
    if not bt_ranked or len(bt_ranked) != len(bases):
        return None

    total = 0.0
    for opts, ib, it in zip(bt_ranked, bases, tones):
        best_rank: Optional[int] = None

        if not restrict_tone:
            # Base-only matching (ignore tone), still prefer common readings by rank.
            for cb, _ct, rank in opts:
                if cb == ib:
                    best_rank = rank if best_rank is None else min(best_rank, rank)
        else:
            if ib == "er":
                # allow er(2/5/0) interchange when input is 2 or 5
                for cb, ct, rank in opts:
                    if cb != "er":
                        continue
                    if it in (2, 5) and ct in (0, 2, 5):
                        best_rank = rank if best_rank is None else min(best_rank, rank)
                    elif ct == it:
                        best_rank = rank if best_rank is None else min(best_rank, rank)
            else:
                if it == 5:
                    for cb, ct, rank in opts:
                        if cb == ib and ct in (5, 0):
                            best_rank = (
                                rank if best_rank is None else min(best_rank, rank)
                            )
                else:
                    for cb, ct, rank in opts:
                        if cb == ib and ct == it:
                            best_rank = (
                                rank if best_rank is None else min(best_rank, rank)
                            )

        if best_rank is None:
            return None
        if best_rank > MAX_PRON_RANK:
            return None

        total += float(best_rank)

    return total


# ----------------------------
# Pinyin2Hanzi decode
# ----------------------------
def _dag_decode(bases: List[str], topk: int) -> List[Tuple[str, float]]:
    if DefaultDagParams is None or dag is None:
        return []
    params = DefaultDagParams()
    try:
        res = dag(params, bases, path_num=topk)
    except TypeError:
        res = dag(params, bases, topk)

    tmp: Dict[str, float] = {}
    for r in res:
        if hasattr(r, "path"):
            hz = "".join(r.path)
        elif hasattr(r, "hanzi"):
            hz = r.hanzi
        elif hasattr(r, "sentence"):
            hz = r.sentence
        else:
            hz = str(r)
        sc = float(getattr(r, "score", 0.0))
        if hz not in tmp or sc > tmp[hz]:
            tmp[hz] = sc
    return sorted(tmp.items(), key=lambda x: x[1], reverse=True)


# ----------------------------
# DAG input compatibility (fix yue-case without breaking others)
# Some Pinyin2Hanzi builds store üe as "ve" after y/j/q/x (and sometimes n/l).
# Your recovered pinyin uses standard spelling ("yue"), so we try both:
#   - first: original bases (no change)
#   - fallback: map {yue,jue,que,xue,nue,lue} -> {yve,jve,qve,xve,nve,lve}
# We still validate STRICTLY against the ORIGINAL bases+tones.
# ----------------------------
_UE_TO_VE_INI = set("yjqxnl")


def _dag_trylists(bases: List[str]) -> List[List[str]]:
    alt = []
    changed = False
    for b in bases:
        if len(b) >= 3 and b[0] in _UE_TO_VE_INI and b[1:] == "ue":
            alt.append(b[0] + "ve")
            changed = True
        else:
            alt.append(b)
    return [bases] if not changed else [bases, alt]


def _lex_penalty(words_hz: List[str]) -> float:
    if not _JIEBA_FREQ:
        return 0.0
    pen = 0.0
    for w in words_hz:
        # FREQ is a dict-like; unknown => 0
        f = _JIEBA_FREQ.get(w, 0) if hasattr(_JIEBA_FREQ, "get") else 0
        if f <= 0:
            # unknown multi-char words get stronger penalty; single char very weak
            pen += 1.2 if len(w) >= 2 else 0.15
        else:
            # small bonus for known words (subtract penalty)
            pen -= min(0.6, 0.12 * math.log(f + 1.0))
    return pen


def _split_hz_by_counts(hz: str, counts: List[int]) -> Optional[List[str]]:
    if sum(counts) != len(hz):
        return None
    out: List[str] = []
    idx = 0
    for c in counts:
        out.append(hz[idx : idx + c])
        idx += c
    return out


def _decode_window_list(
    words: List[str], topk: int, restrict_tone: bool
) -> Optional[List[str]]:
    """
    Decode a list of pinyin words into hanzi words (same word count),
    selecting among STRICT candidates by (pron_rank_penalty + lex_penalty, then LM score).
    """
    bases_all: List[str] = []
    tones_all: List[int] = []
    counts: List[int] = []

    for w in words:
        parsed = _parse_word(w)
        if parsed is None:
            return None
        b, t = parsed
        bases_all.extend(b)
        tones_all.extend(t)
        counts.append(len(b))

        # Try DAG with original bases first; if no strict candidate, fall back to an alternate encoding
    # (fixes cases like "yue" not found in some Pinyin2Hanzi dictionaries).
    best_words: Optional[List[str]] = None
    best_key: Optional[Tuple[float, float]] = None  # (penalty, -lm_score)

    for bases_dag in _dag_trylists(bases_all):
        cands = _dag_decode(bases_dag, topk=topk)
        for hz, lm_sc in cands:
            # IMPORTANT: strict validation always uses ORIGINAL bases_all + tones_all
            pron_pen = _strict_match_penalty(
                bases_all, tones_all, hz, restrict_tone=restrict_tone
            )
            if pron_pen is None:
                continue

            words_hz = _split_hz_by_counts(hz, counts)
            if words_hz is None:
                continue

            pen = pron_pen
            if LEX_W > 0.0:
                pen += LEX_W * _lex_penalty(words_hz)

            key = (pen, -lm_sc)
            if best_key is None or key < best_key:
                best_key = key
                best_words = words_hz

        # if we already found something with the primary encoding, don't waste time
        if best_words is not None:
            break

    return best_words


@functools.lru_cache(maxsize=8192)
def _decode_window_cached(
    words: Tuple[str, ...], topk: int, restrict_tone: bool
) -> Optional[Tuple[str, ...]]:
    res = _decode_window_list(list(words), topk=topk, restrict_tone=restrict_tone)
    return tuple(res) if res is not None else None


def _word_strict_ok(pinyin_word: str, hanzi_word: str, restrict_tone: bool) -> bool:
    parsed = _parse_word(pinyin_word)
    if parsed is None:
        return False
    bases, tones = parsed
    return (
        _strict_match_penalty(bases, tones, hanzi_word, restrict_tone=restrict_tone)
        is not None
    )


def _decode_run_words(run_words: List[str], restrict_tone: bool) -> List[str]:
    """
    Keep word boundaries, but use cross-boundary context:
    1) decode full run when feasible
    2) else overlapping-window voting
    """
    m = len(run_words)
    if m == 0:
        return []

    # 1) full-run decode
    if m <= FULL_RUN_MAX_WORDS:
        full_topk = min(TOPK_CAP, max(TOPK_FULL_BASE, TOPK_FULL_PER_WORD * m))
        dec_full = _decode_window_cached(tuple(run_words), full_topk, restrict_tone)
        if dec_full is not None and len(dec_full) == m:
            return list(dec_full)

    # 2) overlapping-window voting
    votes = [defaultdict(float) for _ in range(m)]
    for start in range(m):
        window = run_words[start : start + WINDOW_WORDS]
        if not window:
            continue
        topk = min(TOPK_CAP, max(TOPK_WINDOW_BASE, 90 * len(window)))
        dec = _decode_window_cached(tuple(window), topk, restrict_tone)
        if dec is None:
            continue

        L = len(window)
        center = (L - 1) / 2.0
        denom = center + 1.0
        for off in range(L):
            idx = start + off
            if idx >= m:
                break
            w = 1.0 - abs(off - center) / denom
            if _word_strict_ok(run_words[idx], dec[off], restrict_tone):
                votes[idx][dec[off]] += w

    out: List[str] = []
    for i in range(m):
        if votes[i]:
            best = max(votes[i].items(), key=lambda kv: kv[1])[0]
            if _word_strict_ok(run_words[i], best, restrict_tone):
                out.append(best)
                continue

        # fallback: single word with a larger topk
        dec1 = _decode_window_cached(
            (run_words[i],), min(TOPK_CAP, max(TOPK_WINDOW_BASE, 500, restrict_tone))
        )
        out.append(dec1[0] if dec1 is not None else run_words[i])

    return out


def suggest_hanzi_text(
    pinyin_text: str, restrict_tone: bool = True, *args, **kwargs
) -> str:
    """
    Public API. Extra args/kwargs accepted for backward compatibility, but ignored.
    """
    if DefaultDagParams is None or dag is None:
        return "[Pinyin2Hanzi 未安装：pip install Pinyin2Hanzi]"

    tokens = [m.group(0) for m in _TOKEN_RE.finditer(pinyin_text)]
    out_tokens = tokens[:]

    # Build runs of word-indices across commas/quotes/etc; only hard breaks split runs
    runs: List[List[int]] = []
    cur: List[int] = []
    for idx, tok in enumerate(tokens):
        if _is_word(tok):
            cur.append(idx)
            continue
        if _is_hard_break(tok):
            if cur:
                runs.append(cur)
                cur = []
            continue
        # other tokens do not end run

    if cur:
        runs.append(cur)

    for run_idxs in runs:
        run_words = [tokens[i] for i in run_idxs]
        decoded = _decode_run_words(run_words, restrict_tone)
        for tok_i, hz in zip(run_idxs, decoded):
            out_tokens[tok_i] = hz

    return "".join(out_tokens)
