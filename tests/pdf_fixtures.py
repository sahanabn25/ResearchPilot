"""Generate FICTIONAL research-paper PDFs for tests and demos (ReportLab). Nothing here is a real study.

make_paper_pdf(...) options cover: single/two columns, IEEE-style headings, missing 'Abstract' heading, running
headers/footers, extra pages, owner-only or password encryption. make_image_only_pdf() makes a 'scanned' PDF.
"""
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

TITLE = "Hybrid CNN-LSTM with Attention for Regional Weather Forecasting"
ABSTRACT = ("We propose a hybrid CNN-LSTM model with an attention layer for 24-hour regional temperature forecasting. "
            "Evaluated on the ERA5 reanalysis dataset, the model reaches an RMSE of 1.1 compared with 1.4 for a "
            "persistence baseline. Results are averaged over 5 random seeds.")
FILLER = [
    "Accurate short-term forecasts support planning in agriculture, energy and transport, and they remain difficult "
    "for regional grids because local effects are hard to capture.",
    "Recent studies apply deep learning to gridded atmospheric data, yet many treat spatial and temporal structure "
    "separately and report results on a single region.",
    "Our approach combines convolutional feature extraction with recurrent sequence modelling so that both kinds of "
    "structure are learned together in one lightweight network.",
    "All experiments were run on a single graphics card and each configuration was trained until the validation loss "
    "stopped improving for several consecutive epochs.",
    "We discuss how the attention weights can be inspected to understand which time steps influence a prediction, "
    "which may help forecasters trust the model.",
    "Data preprocessing followed common practice: variables were normalised, missing values were interpolated and the "
    "final two years were held out as a test set.",
]
CONTENT = {
    "Introduction": [
        "Short-term temperature forecasting is an important problem for many applications. The problem addressed in "
        "this paper is how to improve 24-hour regional temperature forecasts using a compact neural model.",
        "The main contributions of this paper are a hybrid CNN-LSTM architecture with attention and an evaluation "
        "against a persistence baseline on the ERA5 dataset."],
    "Related Work": ["Earlier work used numerical weather prediction and, more recently, recurrent and convolutional "
                     "networks trained on reanalysis data."],
    "Methodology": [
        "The proposed model passes each input grid through convolutional layers and feeds the features to an LSTM with "
        "an attention layer. The objective is to minimise the root mean squared error of the forecast.",
        "RMSE = sqrt( (1/N) * sum_i (y_i - yhat_i)^2 )   (1)",
        "We trained with the Adam optimiser and early stopping."],
    "Experiments": [
        "We use the ERA5 reanalysis dataset and compare against a persistence baseline and a plain LSTM. "
        "Table 1 lists the forecast error on the held-out test set."],
    "Results": [
        "The proposed CNN-LSTM achieves an RMSE of 1.1, lower than the persistence baseline at 1.4 and the plain LSTM "
        "at 1.25. Results are averaged over 5 random seeds."],
    "Conclusion": [
        "In this paper we presented a hybrid CNN-LSTM model for regional temperature forecasting. A limitation of this "
        "study is that only European data was used. In future work we plan to evaluate on other regions and to add "
        "uncertainty estimates."],
}
REFERENCES = ["[1] A. Author. Deep learning for weather prediction. Journal of Examples, 2020.",
              "[2] B. Writer. Attention in sequence models. Proceedings of Examples, 2019.",
              "[3] C. Scholar. Reanalysis data for machine learning. Data Notes, 2021."]
TABLE_ROWS = [["Model", "RMSE", "MAE"], ["Persistence", "1.4", "0.9"], ["LSTM", "1.25", "0.8"], ["CNN-LSTM (ours)", "1.1", "0.7"]]


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20),
        "meta": ParagraphStyle("m", parent=base["Normal"], fontSize=9, leading=12, alignment=1),
        "h": ParagraphStyle("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=10, leading=13),
        "cap": ParagraphStyle("c", parent=base["Normal"], fontSize=9, leading=11, spaceBefore=4, spaceAfter=6),
    }


def _heading(label, number, style_name):
    if style_name == "ieee":
        roman = ["I", "II", "III", "IV", "V", "VI", "VII"][number - 1]
        return f"{roman}. {label.upper()}"
    return f"{number}. {label}"


def _story(opts):
    st = _styles()
    story = [Paragraph(TITLE, st["title"]),
             Paragraph("A. Researcher, B. Scientist", st["meta"]),
             Paragraph("Department of Computer Science, Example University, researcher@example.edu", st["meta"]),
             Spacer(1, 10)]
    ieee = opts.get("ieee_style")
    if opts.get("no_abstract_heading"):
        story.append(Paragraph(ABSTRACT, st["body"]))
    elif ieee:
        story.append(Paragraph(f"<i>Abstract</i>\u2014{ABSTRACT}", st["body"]))
    else:
        story += [Paragraph("Abstract", st["h"]), Paragraph(ABSTRACT, st["body"])]
    heading_style = "ieee" if ieee else "plain"
    for n, (name, paragraphs) in enumerate(CONTENT.items(), 1):
        story.append(Paragraph(_heading(name, n, heading_style), st["h"]))
        for text in paragraphs:
            story.append(Paragraph(text, st["body"]))
            story.append(Spacer(1, 4))
        for i in range(opts.get("filler", 6)):
            story.append(Paragraph(FILLER[(i + n) % len(FILLER)] + " " + FILLER[(i + n + 2) % len(FILLER)], st["body"]))
            story.append(Spacer(1, 4))
        if name == "Experiments":
            story.append(Paragraph("Table 1: Forecast error on the ERA5 test set", st["cap"]))
            table = Table(TABLE_ROWS, hAlign="LEFT")
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                       ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
            story += [table, Spacer(1, 8), Paragraph("Figure 2: Training loss curve for the proposed model.", st["cap"])]
    for i in range(opts.get("extra_pages", 0)):
        story += [PageBreak(), Paragraph(f"Appendix filler page {i}. " + FILLER[i % len(FILLER)] * 8, st["body"])]
    story.append(Paragraph("References" if not ieee else "REFERENCES", st["h"]))
    story += [Paragraph(r, st["body"]) for r in REFERENCES]
    return story


def make_paper_pdf(two_column=False, ieee_style=False, no_abstract_heading=False, header_footer=True, extra_pages=0,
                   filler=6, encrypt=None, metadata_title=None) -> bytes:
    """Return the bytes of a fictional research paper PDF."""
    buffer = io.BytesIO()
    enc = None
    if encrypt == "owner":                       # opens without a password
        enc = StandardEncryption("", ownerPassword="owner-secret", canPrint=1)
    elif encrypt == "user":                      # needs a password to open
        enc = StandardEncryption("open-sesame", ownerPassword="owner-secret", canPrint=1)

    def decorate(canvas, doc):
        if header_footer:
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.drawString(2 * cm, A4[1] - 1.2 * cm, "Journal of Examples 2026")
            canvas.drawCentredString(A4[0] / 2, 1.0 * cm, str(doc.page))
            canvas.restoreState()

    kwargs = dict(pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.2 * cm, bottomMargin=2 * cm,
                  title=metadata_title or "", encrypt=enc)
    opts = dict(ieee_style=ieee_style, no_abstract_heading=no_abstract_heading, extra_pages=extra_pages, filler=filler)
    if two_column:
        doc = BaseDocTemplate(buffer, **kwargs)
        w = (A4[0] - 4 * cm - 0.6 * cm) / 2
        h = A4[1] - 4.2 * cm
        frames = [Frame(2 * cm, 2 * cm, w, h, id="l"), Frame(2 * cm + w + 0.6 * cm, 2 * cm, w, h, id="r")]
        doc.addPageTemplates([PageTemplate(id="two", frames=frames, onPage=decorate)])
        doc.build(_story(opts))
    else:
        SimpleDocTemplate(buffer, **kwargs).build(_story(opts), onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()


def make_image_only_pdf(pages=2) -> bytes:
    """A 'scanned' PDF: pages contain only a raster image, no extractable text."""
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    img = Image.new("RGB", (600, 800), "white")
    for x in range(0, 600, 12):
        for y in range(100, 700, 40):
            img.putpixel((x, y), (0, 0, 0))
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    for _ in range(pages):
        c.drawImage(ImageReader(img), 1 * cm, 2 * cm, width=18 * cm, height=24 * cm)
        c.showPage()
    c.save()
    return buffer.getvalue()


def make_text_pdf(text: str, pages: int = 1) -> bytes:
    """Minimal PDF with the given text on each page (for small edge cases)."""
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    for _ in range(pages):
        y = A4[1] - 2 * cm
        for line in text.splitlines():
            c.drawString(2 * cm, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buffer.getvalue()
