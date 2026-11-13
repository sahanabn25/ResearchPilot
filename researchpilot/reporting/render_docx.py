"""DOCX rendering (python-docx): headings, labelled paragraphs, bullet lists, shaded-header tables, page numbers."""
import io

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from .text import PROVENANCE_COLORS, PROVENANCE_LABELS, clean

DARK, TEAL, GREY = RGBColor(0x12, 0x34, 0x3B), RGBColor(0x2D, 0x6A, 0x6F), RGBColor(0x6B, 0x7C, 0x80)
TEXT_WIDTH_CM = 17.0


def _rgb(hex_color: str) -> RGBColor:
    return RGBColor(int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


def _shade(cell, hex_fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    props.append(shd)


def _repeat_header(row) -> None:
    props = row._tr.get_or_add_trPr()
    props.append(OxmlElement("w:tblHeader"))
    cant = OxmlElement("w:cantSplit")
    props.append(cant)


def _page_number(paragraph) -> None:
    for kind, text in (("begin", None), (None, "PAGE"), ("separate", None), ("end", None)):
        run = paragraph.add_run()
        run.font.size, run.font.color.rgb = Pt(8), GREY
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
        if kind == "separate":                  # placeholder shown until the field updates
            run._r.append(OxmlElement("w:t"))
            run._r[-1].text = "1"


def _tag_run(paragraph, prov, labels=None) -> None:
    labels = labels or PROVENANCE_LABELS
    if prov in labels:
        run = paragraph.add_run(f"[{labels[prov]}] ")
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = _rgb(PROVENANCE_COLORS[prov])


def _tag_line(doc, prov, labels=None) -> None:
    if prov in (labels or PROVENANCE_LABELS):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        _tag_run(p, prov, labels)


def _small(doc, text, italic=True, color=GREY, size=8.5):
    p = doc.add_paragraph()
    run = p.add_run(clean(text))
    run.italic, run.font.size, run.font.color.rgb = italic, Pt(size), color
    p.paragraph_format.space_after = Pt(6)
    return p


def _table(doc, block) -> None:
    n = len(block["header"])
    widths = block.get("widths") or [1] * n
    scale = TEXT_WIDTH_CM / float(sum(widths))
    table = doc.add_table(rows=1, cols=n)
    table.style = "Table Grid"
    table.autofit = False
    for i, col in enumerate(table.columns):      # grid widths (Word and LibreOffice honour these, not just cell widths)
        col.width = Cm(widths[i] * scale)
    total = table._tbl.tblPr.find(qn("w:tblW"))  # explicit total width, otherwise short tables shrink to their content
    if total is not None:
        total.set(qn("w:type"), "dxa")
        total.set(qn("w:w"), str(int(TEXT_WIDTH_CM / 2.54 * 1440)))
    for i, text in enumerate(block["header"]):
        cell = table.rows[0].cells[i]
        cell.width = Cm(widths[i] * scale)
        _shade(cell, "12343B")
        run = cell.paragraphs[0].add_run(clean(text))
        run.bold, run.font.size, run.font.color.rgb = True, Pt(9), RGBColor(255, 255, 255)
    _repeat_header(table.rows[0])
    for r_index, row in enumerate(block["rows"]):
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].width = Cm(widths[i] * scale)
            if r_index % 2:
                _shade(cells[i], "F1F6F5")
            run = cells[i].paragraphs[0].add_run(clean(text))
            run.font.size = Pt(9)
    if block.get("caption"):
        _small(doc, block["caption"])
    else:
        doc.add_paragraph().paragraph_format.space_after = Pt(2)


def to_docx_bytes(report: dict) -> bytes:
    labels = report.get("labels") or PROVENANCE_LABELS
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    for name, size, color in (("Title", 22, DARK), ("Heading 1", 15, DARK), ("Heading 2", 12, TEAL)):
        style = doc.styles[name]
        style.font.name, style.font.size, style.font.color.rgb = "Calibri", Pt(size), color
        style.font.bold = True

    props = doc.core_properties
    props.title, props.author, props.subject = clean(report["title"]), "ResearchPilot", "Research analysis report"

    doc.add_heading(clean(report["title"]), 0)
    _small(doc, report["subtitle"])
    for notice in report.get("notices", []):
        p = doc.add_paragraph()
        run = p.add_run("Notice: " + clean(notice))
        run.italic, run.font.size, run.font.color.rgb = True, Pt(9.5), _rgb("C9731A")

    for section in report["sections"]:
        doc.add_heading(f"{section['number']}. {clean(section['title'])}", level=1)
        for b in section["blocks"]:
            kind = b["type"]
            if kind == "paragraph":
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(6)
                _tag_run(p, b.get("provenance"), labels)
                p.add_run(clean(b["text"]))
            elif kind == "subheading":
                doc.add_heading(clean(b["text"]), level=2)
            elif kind == "note":
                _small(doc, b["text"])
            elif kind == "bullets":
                _tag_line(doc, b.get("provenance"), labels)
                for item in b["items"]:
                    doc.add_paragraph(clean(item), style="List Bullet")
            elif kind == "references":
                for item in b["items"]:
                    p = doc.add_paragraph(clean(item))
                    p.paragraph_format.left_indent = Cm(0.8)
                    p.paragraph_format.first_line_indent = Cm(-0.8)
                    p.paragraph_format.space_after = Pt(3)
                    for run in p.runs:
                        run.font.size = Pt(9.5)
            elif kind == "table":
                _tag_line(doc, b.get("provenance"), labels)
                _table(doc, b)

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    label = footer.add_run("ResearchPilot report  |  page ")
    label.font.size, label.font.color.rgb = Pt(8), GREY
    _page_number(footer)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
