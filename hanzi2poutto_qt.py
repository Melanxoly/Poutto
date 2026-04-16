# hanzi2poutto_qt.py
# pip install pyside6 pypinyin jieba
import re
import sys
import unicodedata

import jieba
from pypinyin import pinyin, Style, load_phrases_dict

from poutto import convert_word as convert_word_latin, syllable_info
from poutto_greek import convert_word_greek
from poutto_deva import convert_word_deva

# reverse modules (you already have these two)
from hanzi import latin_text_to_pinyin
from hanzi_greek import greek_text_to_pinyin

# new deva reverse
from hanzi_deva import deva_text_to_pinyin

# ----------------------------
# Qt compatibility (PySide6 / PyQt6 / PyQt5)
# ----------------------------
QT_API = None
try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QFontMetrics
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
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
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont, QFontMetrics
        from PyQt6.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QTextEdit,
            QCheckBox,
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
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont, QFontMetrics
        from PyQt5.QtWidgets import (
            QApplication,
            QMainWindow,
            QWidget,
            QTextEdit,
            QCheckBox,
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
# Settings (DP splitting for Latin/Greek only in forward mode)
# ----------------------------
MAX_CHARS_PER_SUBWORD = 5
TARGET_LEN = 18
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
# Formatting (forward)
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
    pmap = {"?": ";", ";": "·", ":": "·", "!": "."}
    text = re.sub(r"[?:;!]", lambda m: pmap[m.group(0)], text)
    text = re.sub(r"\s+([,.;·])", r"\1", text)
    text = re.sub(r"([,.;·])(?=[^\s\n])", r"\1 ", text)
    text = re.sub(r"\s*·\s*", " · ", text)
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

    def fix_upper_upsilon_breathing(s: str) -> str:
        """
        After capitalization: for uppercase Υ, keep ONLY rough breathing; remove smooth breathing.
        Exclude ΥΙ where breathing should be on iota.
        Handles both NFC/NFD forms.
        """
        import unicodedata

        # Work in NFD to see combining marks explicitly
        t = unicodedata.normalize("NFD", s)

        out = []
        i = 0
        while i < len(t):
            ch = t[i]

            # Detect Υ + (combining smooth/rough) possibly
            if ch == "Υ":
                # lookahead: next base char to detect ΥΙ
                # but combining marks may be between; we need peek next non-combining after marks
                j = i + 1
                marks = []
                while j < len(t) and unicodedata.combining(t[j]) != 0:
                    marks.append(t[j])
                    j += 1
                next_base = t[j] if j < len(t) else ""

                # If ΥΙ, do nothing here (breathing should sit on iota)
                if next_base in ("Ι", "ι"):
                    out.append(ch)
                    out.extend(marks)
                    i = j
                    continue

                # Otherwise: remove smooth, keep rough
                # smooth breathing = U+0313, rough breathing = U+0314 in combining form
                SMOOTH = "\u0313"
                ROUGH = "\u0314"

                kept = [m for m in marks if m != SMOOTH]  # drop smooth
                # (If both present, rough remains)
                out.append(ch)
                out.extend(kept)
                i = j
                continue

            out.append(ch)
            i += 1

        return unicodedata.normalize("NFC", "".join(out))

    s = fix_upper_upsilon_breathing(s)
    return s


def format_deva_basic(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)

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
        {"贾母": [["jia3"], ["mu3"]], "刘姥姥": [["liu2"], ["lao3"], ["lao5"]]}
    )
    jieba.add_word("转换器", freq=200000)
    jieba.add_word("老刘", freq=200000)
    jieba.add_word("大家子", freq=200000)


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
# DP splitting (forward Latin/Greek only)
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
    seg_text = to_word("-".join(sylls))
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
    for j in range(1, n + 1):
        best = INF
        best_i = -1
        for seg_len in range(1, MAX_CHARS_PER_SUBWORD + 1):
            i = j - seg_len
            if i < 0:
                break
            val = (
                dp[i]
                + _segment_cost(syllables[i:j], to_word)
                + (0.0 if i == 0 else SPLIT_PENALTY)
            )
            if val < best:
                best = val
                best_i = i
        dp[j] = best
        prev[j] = best_i
    segs = []
    cur = n
    while cur > 0:
        i = prev[cur]
        if i < 0:
            return [syllables]
        segs.append(syllables[i:cur])
        cur = i
    segs.reverse()
    return segs


# ----------------------------
# Wrap: only break at spaces (for all outputs)
# ----------------------------
def wrap_only_at_spaces_qt(text: str, font: QFont, max_px: int) -> str:
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
# Scheme dispatch (forward)
# ----------------------------
def get_scheme_forward(scheme: str):
    scheme = scheme.lower()
    if scheme == "latin":
        return (
            convert_word_latin,
            format_english_punc_and_caps,
            QFont("Times New Roman", 16),
        )
    if scheme == "greek":
        return convert_word_greek, format_greek_ancient, QFont("Times New Roman", 16)
    if scheme in ("deva", "devanagari"):
        return convert_word_deva, format_deva_basic, QFont("Kokila", 22)
    raise ValueError(f"Unknown scheme: {scheme}")


# ----------------------------
# Forward: Hanzi -> Scheme text
# ----------------------------
def convert_hanzi(text: str, scheme: str) -> str:
    to_word, post_fmt, _ = get_scheme_forward(scheme)
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
            out.append("".join(PUNC_MAP.get(ch, ch) for ch in tok))
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
        segments = [syllables]
        # if scheme.lower() in ("deva", "devanagari"):
        #     segments = [syllables]
        # else:
        #     segments = split_syllables_dp(syllables, to_word)
        for seg in segments:
            piece = to_word("-".join(seg))
            if prev_was_word:
                out.append(" ")
            out.append(piece)
            prev_was_word = True
    return post_fmt("".join(out))


# ----------------------------
# Reverse: Scheme text -> Pinyin
# ----------------------------
def reverse_to_pinyin(text: str, scheme: str) -> str:
    scheme = scheme.lower()
    if scheme == "latin":
        return latin_text_to_pinyin(text)
    if scheme == "greek":
        return greek_text_to_pinyin(text)
    if scheme in ("deva", "devanagari"):
        return deva_text_to_pinyin(text)
    raise ValueError(f"Unknown scheme: {scheme}")


# ----------------------------
# Qt UI
# ----------------------------
class InputEdit(QTextEdit):
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

    def insertFromMimeData(self, source):
        if source.hasText():
            self.insertPlainText(source.text())
        else:
            super().insertFromMimeData(source)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("POUTTO 转写器")
        self.resize(1150, 800)
        init_resources()

        self.font_in_forward = QFont("SimSun", 14)  # 正向：宋体
        self.font_in_rev_latin = QFont("Times New Roman", 14)
        self.font_in_rev_greek = QFont("Times New Roman", 14)
        self.font_in_rev_deva = QFont("Kokila", 20)  # 天城体稍大一点更舒服
        self.font_pinyin = QFont("Times New Roman", 16)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # Direction row
        dir_row = QHBoxLayout()
        root.addLayout(dir_row)
        dir_row.addWidget(QLabel("方向："))
        self.dir_group = QButtonGroup(self)
        self.rb_forward = QRadioButton("正向：汉字 → 转写")
        self.rb_reverse = QRadioButton("反向：转写 → 拼音")
        self.dir_group.addButton(self.rb_forward)
        self.dir_group.addButton(self.rb_reverse)
        self.rb_forward.setChecked(True)
        self.rb_forward.toggled.connect(self.on_direction_changed)
        self.rb_reverse.toggled.connect(self.on_direction_changed)
        dir_row.addWidget(self.rb_forward)
        dir_row.addWidget(self.rb_reverse)
        dir_row.addStretch(1)

        self.lbl_in = QLabel("输入：")
        root.addWidget(self.lbl_in)

        self.inp = InputEdit(self.do_convert)
        self.inp.setFont(self.font_in_forward)
        self.inp.setStyleSheet("QTextEdit { padding: 10px; }")
        self.inp.setFixedHeight(200)
        self.inp.setPlaceholderText("请在此处输入文本……")
        root.addWidget(self.inp)

        # Controls row
        row = QHBoxLayout()
        root.addLayout(row)
        self.btn_convert = QPushButton("转换 (Ctrl+Enter)")
        self.btn_convert.clicked.connect(self.do_convert)
        row.addWidget(self.btn_convert)
        self.btn_copy = QPushButton("复制输出")
        self.btn_copy.clicked.connect(self.copy_output)
        row.addWidget(self.btn_copy)
        row.addStretch(1)

        self.cb_hanzi = QCheckBox("显示参考汉字")
        self.cb_hanzi.setChecked(False)
        self.cb_hanzi.toggled.connect(self.on_ref_hanzi_toggled)
        self.cb_hanzi.setVisible(False)  # only show in reverse mode
        row.addWidget(self.cb_hanzi)

        row.addWidget(QLabel("方案："))
        self.scheme_group = QButtonGroup(self)
        self.rb_latin = QRadioButton("Latin")
        self.rb_greek = QRadioButton("Greek")
        self.rb_deva = QRadioButton("Devanagari")
        self.scheme_group.addButton(self.rb_latin)
        self.scheme_group.addButton(self.rb_greek)
        self.scheme_group.addButton(self.rb_deva)
        self.rb_latin.setChecked(True)
        self.rb_latin.toggled.connect(self.on_scheme_changed)
        self.rb_greek.toggled.connect(self.on_scheme_changed)
        self.rb_deva.toggled.connect(self.on_scheme_changed)
        row.addWidget(self.rb_latin)
        row.addWidget(self.rb_greek)
        row.addWidget(self.rb_deva)

        self.lbl_out = QLabel("输出：")
        root.addWidget(self.lbl_out)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setStyleSheet("QTextEdit { padding: 12px; }")
        self.out.setLineWrapMode(
            QTextEdit.LineWrapMode.NoWrap
        )  # we manually insert '\n'
        root.addWidget(self.out, stretch=1)

        # Reference Hanzi (hidden unless enabled in reverse mode)
        self.ref_container = QWidget()
        ref_layout = QVBoxLayout(self.ref_container)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        ref_layout.setSpacing(6)

        self.lbl_ref = QLabel("参考汉字（仅供参考）：")
        ref_layout.addWidget(self.lbl_ref)

        self.out_hanzi = QTextEdit()
        self.out_hanzi.setReadOnly(True)
        self.out_hanzi.setFont(QFont("Microsoft YaHei UI", 12))
        self.out_hanzi.setStyleSheet("QTextEdit { padding: 12px; }")
        self.out_hanzi.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        ref_layout.addWidget(self.out_hanzi, stretch=1)

        self.ref_container.setVisible(False)
        root.addWidget(self.ref_container, stretch=1)

        self.refresh_mode_ui(clear_input=False)

    def apply_input_font(self, scheme: str, reverse: bool):
        if not reverse:
            self.inp.setFont(self.font_in_forward)
            return

        if scheme == "deva":
            self.inp.setFont(self.font_in_rev_deva)
        elif scheme == "greek":
            self.inp.setFont(self.font_in_rev_greek)
        else:
            self.inp.setFont(self.font_in_rev_latin)

    def current_scheme(self) -> str:
        if self.rb_deva.isChecked():
            return "deva"
        if self.rb_greek.isChecked():
            return "greek"
        return "latin"

    def is_reverse(self) -> bool:
        return self.rb_reverse.isChecked()

    def apply_output_font(self, scheme: str, reverse: bool):
        if reverse:
            self.out.setFont(self.font_pinyin)
            return
        _, _, f = get_scheme_forward(scheme)
        self.out.setFont(f)

    def refresh_mode_ui(self, clear_input: bool = True):
        """
        Update labels/fonts/visibility for current direction+scheme.
        If clear_input=True, clear the input box (requested on direction switches).
        Output boxes are left unchanged.
        """
        scheme = self.current_scheme()
        reverse = self.is_reverse()

        # Fonts
        self.apply_input_font(scheme, reverse)
        self.apply_output_font(scheme, reverse)

        # Labels + placeholder
        if not reverse:
            self.lbl_in.setText("输入（汉字）：")
            self.lbl_out.setText("输出（转写）：")
            self.inp.setPlaceholderText("请在此处输入文本……")
        else:
            self.lbl_in.setText("输入（转写文本）：")
            self.lbl_out.setText("输出（带声调拼音）：")
            self.inp.setPlaceholderText("请在此处输入文本……")

        # Show checkbox only in reverse mode; checked state is remembered automatically
        self.cb_hanzi.setVisible(reverse)

        # Reference box only when reverse + checked
        show_ref = reverse and self.cb_hanzi.isChecked()
        self.ref_container.setVisible(show_ref)
        if not show_ref:
            self.out_hanzi.clear()

        if clear_input:
            self.inp.clear()

    def on_direction_changed(self, checked: bool):
        # only react when a radio is turned ON
        if not checked:
            return
        # Clear input, keep output unchanged
        self.refresh_mode_ui(clear_input=True)

    def on_scheme_changed(self, checked: bool):
        # only react when a radio is turned ON
        if not checked:
            return

        # Update fonts/labels; keep input text
        self.refresh_mode_ui(clear_input=False)

        # If user has input content, refresh output automatically
        if self.inp.toPlainText().strip():
            self.do_convert()

    def on_ref_hanzi_toggled(self, checked: bool):
        # Only meaningful in reverse mode; state is remembered even when hidden.
        reverse = self.is_reverse()
        self.ref_container.setVisible(reverse and checked)
        if not (reverse and checked):
            self.out_hanzi.clear()
            return

        # If we already have pinyin output, generate reference Hanzi without re-running conversion
        try:
            from hanzi_suggest import suggest_hanzi_text

            pinyin_out = self.out.toPlainText().strip()
            if pinyin_out:
                self.out_hanzi.setPlainText(suggest_hanzi_text(pinyin_out))
        except Exception as e:
            self.out_hanzi.setPlainText(f"[参考汉字生成失败: {type(e).__name__}: {e}]")

    def do_convert(self):
        try:
            scheme = self.current_scheme()
            reverse = self.is_reverse()
            # Ensure UI state is consistent; do not clear input here
            self.refresh_mode_ui(clear_input=False)

            src = self.inp.toPlainText()

            if not reverse:
                self.lbl_in.setText("输入（汉字）：")
                self.lbl_out.setText("输出（转写）：")
                txt = convert_hanzi(src, scheme=scheme)
            else:
                self.lbl_in.setText("输入（转写文本）：")
                self.lbl_out.setText("输出（带声调拼音）：")
                txt = reverse_to_pinyin(src, scheme=scheme)

            # wrap only at spaces for all outputs
            viewport_w = self.out.viewport().width()
            safety = 80
            max_px = max(120, viewport_w - safety)
            font_for_wrap = self.out.font()
            txt = wrap_only_at_spaces_qt(txt, font_for_wrap, max_px)

            self.out.setPlainText(txt)

            # 参考汉字：仅在反向模式且勾选时显示
            if reverse and self.cb_hanzi.isChecked():
                self.ref_container.setVisible(True)
                try:
                    from hanzi_suggest import suggest_hanzi_text

                    hz_ref = suggest_hanzi_text(txt)
                except Exception as e:
                    hz_ref = f"[参考汉字生成失败: {type(e).__name__}: {e}]"
                self.out_hanzi.setPlainText(hz_ref)
            else:
                self.out_hanzi.clear()
                self.ref_container.setVisible(False)

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
