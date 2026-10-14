"""Markdown-subset → ReportLab flowables, plus the document template (TOC, bookmarks, header/footer).

Supported markup in the hand-written chapters:
  #, ##, ###, ####      headings (level 0-3; levels 0-2 appear in the TOC and PDF outline)
  paragraphs            **bold**, *italic*, `code`
  - item / 1. item      bullet / numbered lists (one level, continuation lines indented)
  ```lang ... ```       code block (python / text / bash / yaml highlighted with Pygments)
  | a | b |             pipe tables (first row = header, second row = separator)
  $$ latex $$           display equation rendered with matplotlib mathtext
  ![caption](path)      figure (path relative to the project root)
  > text                callout box
  ---page---            page break
"""
from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygments import lex
from pygments.lexers import BashLexer, PythonLexer, TextLexer, YamlLexer
from pygments.token import Token
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(__file__).resolve().parent / "build"
PAGE_W, PAGE_H = A4
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 18 * mm, 20 * mm, 18 * mm
FRAME_W = PAGE_W - 2 * MARGIN_X

# ---------------------------------------------------------------------------- fonts
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Body", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(_FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", str(_FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(_FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("Mono-Bold", str(_FONT_DIR / "DejaVuSansMono-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Mono-Italic", str(_FONT_DIR / "DejaVuSansMono-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("Mono-BoldItalic", str(_FONT_DIR / "DejaVuSansMono-BoldOblique.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono-Italic", boldItalic="Mono-BoldItalic")

# --------------------------------------------------------------------------- colours
INK = colors.HexColor("#1b1f27")
INK2 = colors.HexColor("#4a5160")
MUTED = colors.HexColor("#7a8191")
ACCENT = colors.HexColor("#2a78d6")
ACCENT_DARK = colors.HexColor("#1c5cab")
RULE = colors.HexColor("#d9dce3")
CODE_BG = colors.HexColor("#f5f6f8")
TABLE_HEAD = colors.HexColor("#e8eef8")
ZEBRA = colors.HexColor("#fafbfc")
CALLOUT_BG = colors.HexColor("#eef4fc")

# ---------------------------------------------------------------------------- styles
S = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Body", fontSize=7.6, leading=10, textColor=INK2),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=7.4, leading=9.4, textColor=INK),
    "cellmono": ParagraphStyle("cellmono", fontName="Mono", fontSize=6.9, leading=8.8, textColor=INK),
    "cellhead": ParagraphStyle("cellhead", fontName="Body-Bold", fontSize=7.4, leading=9.4, textColor=INK),
    "h0": ParagraphStyle("h0", fontName="Body-Bold", fontSize=20, leading=25, textColor=ACCENT_DARK, spaceBefore=4,
                         spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Body-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=11.2, leading=14.5, textColor=INK, spaceBefore=9, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=9.6, leading=12.5, textColor=ACCENT_DARK, spaceBefore=7,
                         spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName="Body", fontSize=9.2, leading=13, textColor=INK, leftIndent=14,
                             bulletIndent=4, spaceAfter=2),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=6.75, leading=8.6, textColor=INK, backColor=CODE_BG,
                           borderPadding=(4, 5, 4, 5), leftIndent=5, rightIndent=5, spaceBefore=4, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="Body-Italic", fontSize=7.8, leading=10, textColor=INK2,
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "callout": ParagraphStyle("callout", fontName="Body", fontSize=8.8, leading=12.4, textColor=INK, backColor=CALLOUT_BG,
                              borderPadding=(6, 8, 6, 8), leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=10,
                              borderColor=ACCENT, borderWidth=0),
    "toc0": ParagraphStyle("toc0", fontName="Body-Bold", fontSize=9.4, leading=13, leftIndent=0, spaceBefore=5),
    "toc1": ParagraphStyle("toc1", fontName="Body", fontSize=8.6, leading=11.4, leftIndent=12),
    "toc2": ParagraphStyle("toc2", fontName="Body", fontSize=7.8, leading=10.2, leftIndent=26, textColor=INK2),
}
CODE_CHARS = int((FRAME_W - 20) / (0.602 * 6.75))  # characters that fit on one code line

# ------------------------------------------------------------------- inline markup


# Characters the embedded DejaVu fonts cannot draw (emoji in UI labels, ℓ in the mono font) are replaced by readable
# names so that nothing renders as a missing-glyph box. Applied to text and to code listings.
GLYPH_FALLBACK = {
    "⏭": "[next]", "⏮": "[prev]", "⏸": "[pause]", "✅": "[✓]", "⬜": "[ ]", "🎓": "[faculty]", "📂": "[folder]",
    "📈": "[chart]", "📊": "[bar chart]", "🔄": "[reset]", "🔍": "[magnifier]", "🚗": "[car]", "ℹ": "(i)",
    "️": "", "⚖": "[scales]",
}


def safe_glyphs(text: str, mono: bool = False) -> str:
    for k, v in GLYPH_FALLBACK.items():
        if k in text:
            text = text.replace(k, v)
    if mono and "ℓ" in text:
        text = text.replace("ℓ", "l")
    return text


def inline(text: str) -> str:
    """Escape and convert **bold**, *italic* and `code` into ReportLab paragraph markup."""
    text = safe_glyphs(text)
    codes: list[str] = []

    def stash(m: re.Match) -> str:
        code = html.escape(safe_glyphs(m.group(0)[1:-1], mono=True), quote=False)
        codes.append(f'<font face="Mono" size="-0.8" color="#1c5cab">{code}</font>')
        return f"\x00{len(codes) - 1}\x00"

    p = re.sub(r"`[^`]+`", stash, text)          # protect code spans, then format the rest as a whole
    p = html.escape(p, quote=False)
    p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
    p = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", p)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], p)


# ------------------------------------------------------------------------- code


_TOKEN_COLORS = [
    (Token.Comment, "#6a737d"), (Token.Literal.String.Doc, "#6a737d"), (Token.Literal.String, "#22863a"),
    (Token.Literal.Number, "#005cc5"), (Token.Keyword, "#d73a49"), (Token.Name.Builtin, "#6f42c1"),
    (Token.Name.Function, "#6f42c1"), (Token.Name.Class, "#6f42c1"), (Token.Name.Decorator, "#e36209"),
    (Token.Operator.Word, "#d73a49"), (Token.Name.Tag, "#22863a"), (Token.Name.Attribute, "#6f42c1"),
]
_LEXERS = {"python": PythonLexer, "py": PythonLexer, "bash": BashLexer, "yaml": YamlLexer, "text": TextLexer, "": TextLexer}


def _color_for(ttype) -> Optional[str]:
    for t, c in _TOKEN_COLORS:
        if ttype in t:
            return c
    return None


def wrap_code_lines(lines: list[str], width: int) -> list[tuple[str, bool]]:
    """Hard-wrap long code lines; returns (text, is_continuation)."""
    out = []
    for line in lines:
        line = line.rstrip("\n").expandtabs(4)
        if len(line) <= width:
            out.append((line, False))
            continue
        indent = len(line) - len(line.lstrip())
        first = True
        while line:
            chunk, line = line[:width], line[width:]
            out.append((chunk, not first))
            first = False
            if line:
                line = " " * min(indent + 4, width // 2) + line
    return out


def code_block(code: str, lang: str = "python", start_line: Optional[int] = None) -> list[Flowable]:
    """Syntax-highlighted code with optional original line numbers (splits across pages)."""
    lines = safe_glyphs(code, mono=True).split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    gutter = 0 if start_line is None else len(str(start_line + len(lines))) + 1
    wrapped_src = wrap_code_lines(lines, CODE_CHARS - gutter)
    lexer = _LEXERS.get(lang, TextLexer)()
    # Highlight line by line (keeps the wrapping logic simple and robust).
    out_lines = []
    lineno = start_line
    for text, cont in wrapped_src:
        tokens = list(lex(text, lexer)) if text else []
        frag = []
        for ttype, value in tokens:
            value = value.rstrip("\n")
            if not value:
                continue
            esc = html.escape(value, quote=False).replace(" ", "&nbsp;")
            c = _color_for(ttype)
            frag.append(f'<font color="{c}">{esc}</font>' if c else esc)
        prefix = ""
        if start_line is not None:
            label = "" if cont else str(lineno)
            prefix = f'<font color="#9aa1ad">{label.rjust(gutter - 1)}&nbsp;</font>'
            if not cont:
                lineno += 1
        if cont:
            prefix += '<font color="#9aa1ad">↪</font>'
        out_lines.append(prefix + "".join(frag))
    return [XPreformatted("\n".join(out_lines), S["code"])]


# ------------------------------------------------------------------------ tables


def table(rows: list[list[str]], col_widths: Optional[list[float]] = None, mono_cols: Iterable[int] = (),
          header: bool = True, font_scale: float = 1.0) -> Table:
    mono_cols = set(mono_cols)
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, text in enumerate(row):
            if header and r == 0:
                cells.append(Paragraph(inline(str(text)), S["cellhead"]))
            elif c in mono_cols:
                cells.append(Paragraph(html.escape(safe_glyphs(str(text), mono=True), quote=False).replace("\n", "<br/>"),
                                       S["cellmono"]))
            else:
                cells.append(Paragraph(inline(str(text)).replace("\n", "<br/>"), S["cell"]))
        data.append(cells)
    ncol = max(len(r) for r in rows)
    if col_widths is None:
        col_widths = [FRAME_W / ncol] * ncol
    else:
        total = sum(col_widths)
        col_widths = [w / total * FRAME_W for w in col_widths]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD))
    for r in range(1 if header else 0, len(rows)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------- equations


def equation(latex: str, fontsize: int = 13) -> Image:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(latex.encode()).hexdigest()[:12]
    path = BUILD_DIR / f"eq_{key}.png"
    if not path.exists():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=fontsize, color="#1b1f27")
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)
    img = Image(str(path))
    scale = min(1.0, (FRAME_W - 20) / img.imageWidth * 300 / 72) if img.imageWidth else 1.0
    img.drawWidth = img.imageWidth * 72 / 300 * scale
    img.drawHeight = img.imageHeight * 72 / 300 * scale
    img.hAlign = "CENTER"
    return img


def figure(path: Path, caption: str = "", max_height: float = 175 * mm, width: Optional[float] = None) -> list[Flowable]:
    img = Image(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    w = width or FRAME_W
    h = w * ih / iw
    if h > max_height:
        h = max_height
        w = h * iw / ih
    img.drawWidth, img.drawHeight = w, h
    img.hAlign = "CENTER"
    flows: list[Flowable] = [img]
    if caption:
        flows.append(Paragraph(inline(caption), S["caption"]))
    return [KeepTogether(flows)]


# ----------------------------------------------------------------------- headings


class Heading(Paragraph):
    """Paragraph that registers itself in the TOC and PDF outline."""

    def __init__(self, text: str, level: int, key: Optional[str] = None) -> None:
        super().__init__(inline(text), S[f"h{level}"])
        self.level = level
        self.plain = re.sub(r"[`*]", "", text)
        self.key = key or "h" + hashlib.md5(f"{text}{id(self)}".encode()).hexdigest()[:10]


def heading(text: str, level: int) -> list[Flowable]:
    flows: list[Flowable] = []
    if level == 0:
        flows.append(PageBreak())
    elif level == 1:
        flows.append(CondPageBreak(60 * mm))
    elif level == 2:
        flows.append(CondPageBreak(35 * mm))
    flows.append(Heading(text, level))
    return flows


# ------------------------------------------------------------------- markdown


def markdown(text: str) -> list[Flowable]:
    lines = text.strip("\n").split("\n")
    flows: list[Flowable] = []
    i = 0
    para: list[str] = []
    widths: Optional[list[float]] = None  # set by a "<!-- widths: 1,3 -->" line, used by the next table

    def flush() -> None:
        if para:
            flows.append(Paragraph(inline(" ".join(s.strip() for s in para)), S["body"]))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            flows += code_block("\n".join(buf), lang or "text")
            i = j + 1
            continue
        m = re.match(r"^<!--\s*widths:\s*([\d.,\s]+)-->$", stripped)
        if m:
            flush()
            widths = [float(w) for w in m.group(1).split(",") if w.strip()]
            i += 1
            continue
        if stripped == "---page---":
            flush()
            flows.append(PageBreak())
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush()
            flows += heading(m.group(2), len(m.group(1)) - 1)
            i += 1
            continue
        if stripped.startswith("$$"):
            flush()
            body = stripped[2:]
            j = i
            while not body.rstrip().endswith("$$"):
                j += 1
                body += " " + lines[j].strip()
            flows.append(Spacer(1, 2))
            flows.append(equation(body.strip()[:-2].strip()))
            flows.append(Spacer(1, 4))
            i = j + 1
            continue
        m = re.match(r"^!\[(.*)\]\((.*?)\)(?:\{w=([\d.]+)\})?$", stripped)  # optional {w=0.8}: fraction of the text width
        if m:
            flush()
            flows += figure(PROJECT_ROOT / m.group(2), m.group(1), width=FRAME_W * float(m.group(3)) if m.group(3) else None)
            i += 1
            continue
        if stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row):
                    rows.append(row)
                i += 1
            flows.append(table(rows, widths))
            flows.append(Spacer(1, 6))
            widths = None
            continue
        if stripped.startswith(">"):
            flush()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            # keep a highlighted box on one page instead of splitting it
            flows.append(KeepTogether([Paragraph(inline(" ".join(buf)), S["callout"])]))
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m and not stripped.startswith("**"):
            flush()
            marker = m.group(2)
            content = [m.group(3)]
            i += 1
            while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and \
                    not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                content.append(lines[i].strip())
                i += 1
            bullet = "•" if marker in "-*" else marker
            indent = len(m.group(1))
            st = S["bullet"]
            if indent:
                st = ParagraphStyle("b2", parent=st, leftIndent=28, bulletIndent=18)
            flows.append(Paragraph(inline(" ".join(content)), st, bulletText=bullet))
            continue
        if not stripped:
            flush()
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    return flows


# --------------------------------------------------------------------- template


class HandoverDoc(BaseDocTemplate):
    def __init__(self, path: Path, title: str, footer: str = "multi_sensor_fusion · technical handover",
                 toc_levels: int = 1, **kw) -> None:
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X, topMargin=MARGIN_TOP,
                         bottomMargin=MARGIN_BOTTOM, title=title, author="Project handover", **kw)
        self.doc_title = title
        self.footer = footer
        self.toc_levels = toc_levels  # deepest heading level listed in the table of contents
        frame = Frame(MARGIN_X, MARGIN_BOTTOM, FRAME_W, PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        # The header is drawn at the *end* of each page so it shows the chapter that starts on that page.
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self._cover),
                               PageTemplate("normal", [frame], onPageEnd=self._page)])
        self._chapter = ""

    def _cover(self, canv, doc) -> None:
        canv.saveState()
        canv.setFillColor(ACCENT_DARK)
        canv.rect(0, PAGE_H - 100 * mm, PAGE_W, 100 * mm, stroke=0, fill=1)
        canv.restoreState()

    def _page(self, canv, doc) -> None:
        canv.saveState()
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.5)
        canv.line(MARGIN_X, PAGE_H - 13 * mm, PAGE_W - MARGIN_X, PAGE_H - 13 * mm)
        canv.setFont("Body", 7.2)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN_X, PAGE_H - 11 * mm, self.doc_title)
        canv.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 11 * mm, self._chapter[:70])
        canv.line(MARGIN_X, 12 * mm, PAGE_W - MARGIN_X, 12 * mm)
        canv.drawRightString(PAGE_W - MARGIN_X, 8 * mm, f"Page {doc.page}")
        canv.drawString(MARGIN_X, 8 * mm, self.footer)
        canv.restoreState()

    def beforeDocument(self) -> None:
        # multiBuild runs several passes; reset per-pass state.
        self._last_outline = -1
        self._chapter = ""

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Heading):
            level = flowable.level
            if level == 0:
                self._chapter = flowable.plain
            key = flowable.key
            self.canv.bookmarkPage(key)
            # PDF outlines cannot skip levels (e.g. chapter → level-2 heading): clamp to previous + 1.
            outline_level = min(level, getattr(self, "_last_outline", -1) + 1, 3)
            self._last_outline = outline_level
            self.canv.addOutlineEntry(flowable.plain, key, level=outline_level, closed=outline_level >= 1)
            if level <= self.toc_levels and getattr(flowable, "in_toc", True):  # deeper levels → PDF bookmarks only
                self.notify("TOCEntry", (level, flowable.plain, self.page, key))


def toc() -> TableOfContents:
    t = TableOfContents()
    t.levelStyles = [S["toc0"], S["toc1"], S["toc2"]]
    t.dotsMinLevel = 0
    return t
