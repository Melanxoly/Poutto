# poutto_qt.py
# pip install pyside6 pypinyin jieba
import re
import sys
import unicodedata

import jieba
from pypinyin import pinyin, Style, load_phrases_dict

from poutto import convert_word as convert_word_latin, syllable_info
from poutto_greek import convert_word_greek
from poutto_deva import convert_word_deva

# ----------------------------
# Qt compatibility (PySide6 / PyQt6 / PyQt5)
# ----------------------------
QT_API = None
try:
    from PySide6.QtCore import Qt, QEvent
    from PySide6.QtGui import QFont, QFontMetrics, QKeySequence
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QTextEdit,
        QPushButton,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QRadioButton,
        QButtonGroup,
        QMessageBox,
    )

    QT_API = "PySide6"
except Exception:
    try:
        from PyQt6.QtCore import Qt, QEvent
        from PyQt6.QtGui import QFont, QFontMetrics, QKeySequence
        from PyQt6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QTextEdit,
            QPushButton,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QRadioButton,
            QButtonGroup,
            QMessageBox,
        )

        QT_API = "PyQt6"
    except Exception:
        from PyQt5.QtCore import Qt, QEvent
        from PyQt5.QtGui import QFont, QFontMetrics, QKeySequence
        from PyQt5.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QTextEdit,
            QPushButton,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QRadioButton,
            QButtonGroup,
            QMessageBox,
        )

        QT_API = "PyQt5"


# ----------------------------
# Settings (DP splitting for Latin/Greek only)
# ----------------------------
MAX_CHARS_PER_SUBWORD = 4
TARGET_LEN = 15
SPLIT_PENALTY = 1.0
LEN_OVER_W = 0.4
LEN_UNDER_W = 0.02

NASAL_W = 1.0
TONE_W = {2: 0.7, 3: 1.0, 4: 0.85}
CLUSTER_W = 0.8
FIVE_CHAR_PENALTY = 2.2

# punctuation detection & mapping (CJK -> western first)
_PUNC_RE = re.compile(r"^[，。！？；：、,.!?;:()（）“”‘’\"\'—\-…]+$")
PUNC_MAP = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ";",
    "：": ":",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "（": "(",
    "）": ")",
    "、": ",",
}


def _is_punc(tok: str) -> bool:
    return bool(_PUNC_RE.fullmatch(tok))


# ----------------------------
# Formatting
# ----------------------------
def format_english_punc_and_caps(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)

    s = re.sub(r"\s+([,\.!?;:])", r"\1", s)
    s = re.sub(r"([,;:])(?=[^\s\n])", r"\1 ", s)
    s = re.sub(r"([\.!?])(?=[^\s\n])", r"\1 ", s)

    s = re.sub(r"(?<!^)(?<!\s)\(", " (", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\)(?=[A-Za-z0-9])", ") ", s)

    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s).strip()

    s = s.replace("……", "…")
    s = s.replace("...", "…")

    chars = list(s)
    cap_next = True
    for i, ch in enumerate(chars):
        if cap_next and "a" <= ch <= "z":
            chars[i] = ch.upper()
            cap_next = False
            continue
        if ch in ".!?":
            cap_next = True
        elif ch.isalpha():
            cap_next = False
        elif ch == "\n":
            cap_next = True
    return "".join(chars)


def format_greek_ancient(text: str) -> str:
    # punctuation mapping
    pmap = {"?": ";", ";": "·", ":": "·", "!": "."}
    text = re.sub(r"[?:;!]", lambda m: pmap[m.group(0)], text)

    # spacing
    text = re.sub(r"\s+([,.;·])", r"\1", text)
    text = re.sub(r"([,.;·])(?=[^\s\n])", r"\1 ", text)
    text = re.sub(r"\s*·\s*", " · ", text)

    # normalize
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()

    QUOTES = set("\"'“”‘’")
    out = []
    cap_next = True
    in_quote = False

    for ch in text:
        if ch in QUOTES:
            if out and out[-1] not in (" ", "\n"):
                out.append(" ")
            if not in_quote:
                cap_next = True
                in_quote = True
            else:
                cap_next = False
                in_quote = False
            continue

        if ch == "\n":
            out.append(ch)
            cap_next = True
            continue

        if cap_next and ch.isalpha():
            out.append(ch.upper())
            cap_next = False
            continue

        out.append(ch)

    s = "".join(out)
    s = s.replace("……", "…")
    s = s.replace("...", "…")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s).strip()
    s = re.sub(r"\s+([,.;·])", r"\1", s)
    s = re.sub(r"\s*·\s*", " · ", s)
    return s


def format_deva_basic(s: str) -> str:
    # Normalize whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)

    # Quote normalization:
    # Convert ASCII quotes/single quotes and CJK quotes into “ ”
    # Strategy: treat any opening quote as “ and any closing quote as ”
    # We'll do a simple toggle pass.
    QUOTES = set(['"', "“", "”"])
    INQUOTE_QUOTES = set(["‘", "’", "'"])
    out = []
    in_quote = False
    in_inquote_quote = False
    for ch in s:
        if ch in QUOTES:
            if not in_quote:
                out.append("“")
                in_quote = True
            else:
                out.append("”")
                in_quote = False
        elif ch in INQUOTE_QUOTES:
            if not in_inquote_quote:
                out.append("‘")
                in_inquote_quote = True
            else:
                out.append("’")
                in_inquote_quote = False
        else:
            out.append(ch)
    s = "".join(out)

    # Ellipsis: keep as … or map; here we map to danda for readability
    s = s.replace("……", "…")
    s = s.replace("...", "…")

    # Sentence end:
    s = s.replace(".", "।")
    s = s.replace(";", "।")
    s = s.replace(":", "।")

    # Optional: keep '?' as modern Hindi question mark
    # (If you want classical Sanskrit style, map '?' -> '।' here.)
    # s = s.replace("?", "।")

    # Spacing rules:
    # - no space before punctuation
    s = re.sub(r"\s+([,।॥?:;”])", r"\1", s)
    # - space after punctuation when followed by non-space/non-newline
    s = re.sub(r"([,।॥?:;“])(?=[^\s\n])", r"\1 ", s)

    # Clean up multiple spaces
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s).strip()

    # Paragraph end: convert final danda at end of each paragraph into double danda
    # (Only if paragraph already ends with a danda-like sentence end.)
    paras = []
    for para in s.split("\n"):
        p = para.rstrip()
        if p.endswith("।"):
            p = p[:-1] + "॥"
        paras.append(p)
    s = "\n".join(paras)

    return unicodedata.normalize("NFC", s)


# ----------------------------
# jieba fixes / resources
# ----------------------------
def init_resources():
    load_phrases_dict(
        {
            "贾母": [["jia3"], ["mu3"]],
            "刘姥姥": [["liu2"], ["lao3"], ["lao5"]],
        }
    )
    # helpful words
    jieba.add_word("转换器", freq=200000)
    jieba.add_word("老刘", freq=200000)


def _fix_common_bad_splits(tokens):
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "老刘老刘":
            out.extend(["老刘", "老刘"])
            i += 1
            continue
        if (
            i + 2 < len(tokens)
            and tokens[i] == "老"
            and tokens[i + 1] == "刘老"
            and tokens[i + 2] == "刘"
        ):
            out.extend(["老刘", "老刘"])
            i += 3
            continue
        out.append(tokens[i])
        i += 1
    return out


# ----------------------------
# DP splitting (Latin/Greek only)
# ----------------------------
def _syllable_risk(pin: str) -> float:
    info = syllable_info(pin)
    tone = info["tone"]
    r = 0.0
    if info["pinyin"] == "yo":
        r += 0.8
    if info["is_nasal"]:
        r += NASAL_W
    if tone in TONE_W:
        r += TONE_W[tone]
    return r


def _segment_cost(sylls: list[str], to_word) -> float:
    seg_len = len(sylls)
    joined = "-".join(sylls)
    seg_text = to_word(joined)
    L = len(seg_text)

    over = max(0, L - TARGET_LEN)
    under = max(0, TARGET_LEN * 0.45 - L)
    length_cost = (over * over) * LEN_OVER_W + (under * under) * LEN_UNDER_W

    size_cost = FIVE_CHAR_PENALTY if seg_len == 5 else 0.0

    risks = [_syllable_risk(s) for s in sylls]
    risk_sum = sum(risks)
    risky_count = sum(1 for r in risks if r > 0.0)
    cluster = 0.0
    if seg_len >= 2 and risky_count >= 2:
        density = risky_count / seg_len
        cluster = (risky_count - 1) ** 2 * density * CLUSTER_W

    return length_cost + 0.35 * risk_sum + cluster + size_cost


def split_syllables_dp(syllables: list[str], to_word) -> list[list[str]]:
    n = len(syllables)
    if n == 0:
        return []

    INF = 1e18
    dp = [INF] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = 0.0
    prev[0] = 0

    for j in range(1, n + 1):
        best = INF
        best_i = -1
        for seg_len in range(1, MAX_CHARS_PER_SUBWORD + 1):
            i = j - seg_len
            if i < 0:
                break
            seg = syllables[i:j]
            val = (
                dp[i] + _segment_cost(seg, to_word) + (0.0 if i == 0 else SPLIT_PENALTY)
            )
            if val < best:
                best = val
                best_i = i
        dp[j] = best
        prev[j] = best_i

    segments = []
    cur = n
    while cur > 0:
        i = prev[cur]
        if i < 0:
            return [syllables]
        segments.append(syllables[i:cur])
        cur = i
    segments.reverse()
    return segments


# ----------------------------
# Devanagari wrapping: only break at spaces
# ----------------------------
def wrap_only_at_spaces_qt(text: str, font: QFont, max_px: int) -> str:
    """
    Insert '\n' only at spaces so that each line <= max_px when possible.
    A single word longer than max_px is NOT split.
    """
    if max_px <= 50:
        return text

    fm = QFontMetrics(font)
    space_w = fm.horizontalAdvance(" ")
    out_lines = []

    for para in text.split("\n"):
        if not para.strip():
            out_lines.append("")
            continue

        words = [w for w in para.split(" ") if w != ""]
        cur_words = []
        cur_w = 0

        for w in words:
            w_w = fm.horizontalAdvance(w)
            if not cur_words:
                cur_words = [w]
                cur_w = w_w
                continue

            cand = cur_w + space_w + w_w
            if cand <= max_px:
                cur_words.append(w)
                cur_w = cand
            else:
                out_lines.append(" ".join(cur_words))
                cur_words = [w]
                cur_w = w_w

        if cur_words:
            out_lines.append(" ".join(cur_words))

    return "\n".join(out_lines)


# ----------------------------
# Scheme dispatch
# ----------------------------
def get_scheme(scheme: str):
    scheme = scheme.lower()
    if scheme == "latin":
        return convert_word_latin, format_english_punc_and_caps
    if scheme == "greek":
        return convert_word_greek, format_greek_ancient
    if scheme in ("deva", "devanagari"):
        return convert_word_deva, format_deva_basic
    raise ValueError(f"Unknown scheme: {scheme}")


# ----------------------------
# Main conversion (shared)
# ----------------------------
def convert_hanzi(text: str, scheme: str) -> str:
    to_word, post_fmt = get_scheme(scheme)

    tokens = _fix_common_bad_splits(jieba.lcut(text))

    out = []
    prev_was_word = False

    for tok in tokens:
        if tok == "\n":
            out.append(tok)
            prev_was_word = False
            continue

        if tok.isspace():
            out.append(tok)
            prev_was_word = False
            continue

        if _is_punc(tok):
            tok = "".join(PUNC_MAP.get(ch, ch) for ch in tok)
            out.append(tok)
            prev_was_word = False
            continue

        if re.fullmatch(r"[A-Za-z0-9_]+", tok):
            if prev_was_word:
                out.append(" ")
            out.append(tok)
            prev_was_word = True
            continue

        pys = pinyin(tok, style=Style.TONE3, neutral_tone_with_five=True, strict=False)
        syllables = [x[0] for x in pys]

        # ✅ Devanagari: no DP splitting
        if scheme.lower() in ("deva", "devanagari"):
            segments = [syllables]
        else:
            segments = split_syllables_dp(syllables, to_word)

        for seg in segments:
            try:
                piece = to_word("-".join(seg))
            except Exception:
                piece = "[" + "-".join(seg) + "]"
            if prev_was_word:
                out.append(" ")
            out.append(piece)
            prev_was_word = True

    return post_fmt("".join(out))


# ----------------------------
# Qt UI
# ----------------------------
class InputEdit(QTextEdit):
    """Capture Ctrl+Enter to trigger convert."""

    def __init__(self, on_convert, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_convert = on_convert

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._on_convert()
            return
        super().keyPressEvent(event)

    # ✅ 新增：粘贴时只插入纯文本
    def insertFromMimeData(self, source):
        if source.hasText():
            self.insertPlainText(source.text())
        else:
            super().insertFromMimeData(source)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("POUTTO 转写器：Latin / Greek / Devanagari (Qt)")
        self.resize(1100, 760)

        init_resources()

        # Fonts per scheme (your request)
        self.font_out_latin = QFont("Times New Roman", 16)
        self.font_out_greek = QFont("Times New Roman", 16)
        self.font_out_deva = QFont("Kokila", 22)

        font_in = QFont("Simsun", 14)

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        root.addWidget(QLabel("输入汉字（可含标点）："))

        self.inp = InputEdit(self.do_convert)
        self.inp.setFont(font_in)
        self.inp.setPlaceholderText("在这里输入汉字…")
        self.inp.setStyleSheet("QTextEdit { padding: 10px; }")
        self.inp.setFixedHeight(180)
        self.inp.setPlainText("请在此处输入汉字文本……")
        root.addWidget(self.inp)

        # Controls row
        row = QHBoxLayout()
        root.addLayout(row)

        self.btn_convert = QPushButton("转写 (Ctrl+Enter)")
        self.btn_convert.clicked.connect(self.do_convert)
        row.addWidget(self.btn_convert)

        self.btn_copy = QPushButton("复制输出")
        self.btn_copy.clicked.connect(self.copy_output)
        row.addWidget(self.btn_copy)

        row.addStretch(1)
        row.addWidget(QLabel("方案："))

        self.group = QButtonGroup(self)
        self.rb_latin = QRadioButton("Latin")
        self.rb_greek = QRadioButton("Greek")
        self.rb_deva = QRadioButton("Devanagari")
        self.group.addButton(self.rb_latin)
        self.group.addButton(self.rb_greek)
        self.group.addButton(self.rb_deva)
        self.rb_latin.setChecked(True)

        self.rb_latin.toggled.connect(self.do_convert)
        self.rb_greek.toggled.connect(self.do_convert)
        self.rb_deva.toggled.connect(self.do_convert)

        row.addWidget(self.rb_latin)
        row.addWidget(self.rb_greek)
        row.addWidget(self.rb_deva)

        root.addWidget(QLabel("输出："))

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setStyleSheet("QTextEdit { padding: 12px; }")
        # For Deva: we will insert manual newlines; so use NoWrap to avoid word-internal breaks
        self.out.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        root.addWidget(self.out, stretch=1)

        self._last_raw = ""
        self._last_scheme = "latin"

        self.do_convert()

    def current_scheme(self) -> str:
        if self.rb_deva.isChecked():
            return "deva"
        if self.rb_greek.isChecked():
            return "greek"
        return "latin"

    def apply_output_font(self, scheme: str):
        if scheme == "deva":
            self.out.setFont(self.font_out_deva)
        elif scheme == "greek":
            self.out.setFont(self.font_out_greek)
        else:
            self.out.setFont(self.font_out_latin)

    def do_convert(self):
        try:
            scheme = self.current_scheme()
            self.apply_output_font(scheme)

            src = self.inp.toPlainText()
            txt = convert_hanzi(src, scheme=scheme)

            # Devanagari: wrap only at spaces; never split words
            viewport_w = self.out.viewport().width()
            safety = 20
            max_px = max(120, viewport_w - safety)

            if scheme == "deva":
                txt = wrap_only_at_spaces_qt(txt, self.font_out_deva, max_px)
            elif scheme == "greek":
                txt = wrap_only_at_spaces_qt(txt, self.font_out_greek, max_px)
            else:
                txt = wrap_only_at_spaces_qt(txt, self.font_out_latin, max_px)

            self._last_raw = txt
            self._last_scheme = scheme
            self.out.setPlainText(txt)

        except Exception as e:
            QMessageBox.critical(self, "转换失败", f"{type(e).__name__}: {e}")

    def copy_output(self):
        QApplication.clipboard().setText(self.out.toPlainText())


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
