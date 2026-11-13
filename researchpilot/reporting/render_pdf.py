"""PDF rendering (ReportLab). Uses a Unicode TrueType font when one is installed (DejaVu, Arial); otherwise
falls back to the built-in Helvetica with Latin-1-safe text, so rendering never fails on unusual characters."""
import io
import os
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .text import PROVENANCE_COLORS, PROVENANCE_LABELS, clean, latin1_safe

DARK, TEAL, GREY, LINE, ZEBRA = "#12343B", "#2D6A6F", "#6B7C80", "#C9D8D5", "#F1F6F5"
_FONT_SETS = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\ariali.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
     "/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf")]
_cache = {}


def _find_font_files():
    """First complete (regular, bold, italic) set present on this machine, else None. Patched in tests."""
    for files in _FONT_SETS:
        if all(os.path.exists(f) for f in files):
            return files
    return None


def _fonts():
    """(regular, bold, italic, unicode_ok)."""
    files = _find_font_files()
    key = files or "builtin"
    if key not in _cache:
        if files:
            try:
                for name, path in zip(("RP-Regular", "RP-Bold", "RP-Italic"), files):
                    pdfmetrics.registerFont(TTFont(name, path))
                pdfmetrics.registerFontFamily("RP-Regular", normal="RP-Regular", bold="RP-Bold",
                                              italic="RP-Italic", boldItalic="RP-Bold")
                _cache[key] = ("RP-Regular", "RP-Bold", "RP-Italic", True)
            except Exception:
                _cache[key] = ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", False)
        else:
            _cache[key] = ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", False)
    return _cache[key]


def to_pdf_bytes(report: dict) -> bytes:
    regular, bold, italic, unicode_ok = _fonts()

    def prep(text) -> str:
        t = clean(text)
        return escape(t if unicode_ok else latin1_safe(t))

    def style(name, **kw):
        base = dict(fontName=regular, fontSize=10, leading=14, textColor=colors.HexColor("#1B2A2F"), alignment=TA_LEFT)
        base.update(kw)
        return ParagraphStyle(name, **base)

    s_title = style("title", fontName=bold, fontSize=20, leading=25, textColor=colors.HexColor(DARK), spaceAfter=4)
    s_sub = style("sub", fontSize=9, leading=12, textColor=colors.HexColor(GREY), spaceAfter=8)
    s_notice = style("notice", fontName=italic, fontSize=9, leading=12, textColor=colors.HexColor("#C9731A"), spaceAfter=3)
    s_h1 = style("h1", fontName=bold, fontSize=14, leading=18, textColor=colors.HexColor(DARK), spaceBefore=14, spaceAfter=6, keepWithNext=1)
    s_h2 = style("h2", fontName=bold, fontSize=11, leading=14, textColor=colors.HexColor(TEAL), spaceBefore=8, spaceAfter=3, keepWithNext=1)
    s_body = style("body", spaceAfter=6)
    s_note = style("note", fontName=italic, fontSize=8.5, leading=11, textColor=colors.HexColor(GREY), spaceAfter=6)
    s_bullet = style("bullet", leftIndent=14, bulletIndent=3, spaceAfter=3)
    s_ref = style("ref", fontSize=9, leading=12, leftIndent=18, firstLineIndent=-18, spaceAfter=3)
    s_tag = style("tag", fontName=bold, fontSize=8, leading=10, spaceAfter=2, keepWithNext=1)
    s_cell = style("cell", fontSize=8, leading=10)
    s_head = style("head", fontName=bold, fontSize=8, leading=10, textColor=colors.white)

    labels = report.get("labels") or PROVENANCE_LABELS

    def tag(prov) -> str:
        if prov not in labels:
            return ""
        return f'<font color="#{PROVENANCE_COLORS[prov]}"><b>[{prep(labels[prov])}]</b></font> '

    avail = A4[0] - 4 * cm
    story = [Paragraph(prep(report["title"]), s_title), Paragraph(prep(report["subtitle"]), s_sub)]
    story += [Paragraph("Notice: " + prep(n), s_notice) for n in report.get("notices", [])]

    for section in report["sections"]:
        story.append(Paragraph(prep(f"{section['number']}. {section['title']}"), s_h1))
        for b in section["blocks"]:
            kind = b["type"]
            if kind == "paragraph":
                story.append(Paragraph(tag(b.get("provenance")) + prep(b["text"]), s_body))
            elif kind == "subheading":
                story.append(Paragraph(prep(b["text"]), s_h2))
            elif kind == "note":
                story.append(Paragraph(prep(b["text"]), s_note))
            elif kind == "bullets":
                if tag(b.get("provenance")):
                    story.append(Paragraph(tag(b["provenance"]), s_tag))
                story += [Paragraph(prep(i), s_bullet, bulletText="\u2022") for i in b["items"]]
            elif kind == "references":
                story += [Paragraph(prep(i), s_ref) for i in b["items"]]
            elif kind == "table":
                widths = b.get("widths") or [1] * len(b["header"])
                col_w = [w / float(sum(widths)) * avail for w in widths]
                data = [[Paragraph(prep(h), s_head) for h in b["header"]]]
                data += [[Paragraph(prep(c), s_cell) for c in row] for row in b["rows"]]
                table = Table(data, colWidths=col_w, repeatRows=1)
                cmds = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(DARK)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(LINE)), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
                cmds += [("BACKGROUND", (0, r), (-1, r), colors.HexColor(ZEBRA)) for r in range(2, len(data), 2)]
                table.setStyle(TableStyle(cmds))
                lead = [Paragraph(tag(b.get("provenance")), s_tag)] if tag(b.get("provenance")) else []
                story += lead + [table]
                story.append(Paragraph(prep(b["caption"]), s_note) if b.get("caption") else Spacer(1, 6))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(regular, 8)
        canvas.setFillColor(colors.HexColor(GREY))
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"ResearchPilot report  |  page {doc.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                      title=clean(report["title"]), author="ResearchPilot").build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
