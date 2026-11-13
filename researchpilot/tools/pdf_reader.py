"""Read an uploaded research-paper PDF: validate, extract text, find sections, tables, equations, figure captions.

Design notes
- Uses pypdf for text (pure Python, MIT) and pdfplumber for tables when it is installed (optional; a text-layout
  heuristic is the fallback). No OCR: a scanned PDF is rejected with a clear message rather than guessed at.
- Everything extracted here is UNTRUSTED text; callers must treat it as data (the parser strips delimiters).
- Limits (size, pages, time) keep a huge or hostile file from freezing the app. Problems become `warnings` or a
  PdfError with a user-facing message; nothing here raises raw library exceptions.
- Equations are 'equation-like lines' recovered from text (formatting is lost); figures are captions only,
  because the data inside graphs is not available without image analysis. The warnings say so.
"""
import io
import logging
import re
import time
from collections import Counter
from typing import Optional

from .. import config
from ..logging_setup import get_logger

log = get_logger(__name__)
for _noisy in ("pypdf", "pdfminer", "pdfplumber"):      # damaged PDFs make these libraries log a lot of warnings
    logging.getLogger(_noisy).setLevel(logging.ERROR)


class PdfError(ValueError):
    """Upload problem with a message that is safe and useful to show to the user."""


MAX_SECTION_CHARS = 30_000
_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl", "\u00ad": "",
              "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_HEADINGS = {
    "abstract": r"abstract|summary",
    "introduction": r"introduction",
    "related_work": r"related work|background|literature review|prior work|related research",
    "methodology": r"methodology|methods?|materials and methods|methods and materials|proposed "
                   r"(?:method|approach|model|framework|system)|approach|model",
    "results": r"experiments?|experimental (?:setup|results|evaluation|design)|results(?: and discussion)?|"
               r"evaluation|empirical (?:results|evaluation)|performance (?:evaluation|analysis)",
    "discussion": r"discussion",
    "limitations": r"limitations?(?: and future (?:work|directions))?",
    "future_work": r"future (?:work|directions|research)",
    "conclusion": r"conclusions?(?: and future (?:work|directions))?|concluding remarks",
    "references": r"references|bibliography",
    "acknowledgements": r"acknowledge?ments?",
}
_NUMBERING = r"(?:(?:\d+(?:\.\d+)*|[IVXivx]+)[\.\)]?\s+)?"
_HEADING_RES = {k: re.compile(rf"^\s*{_NUMBERING}(?:{v})\s*[:\.\-\u2014\u2013]?\s*$", re.IGNORECASE)
                for k, v in _HEADINGS.items()}
_INLINE_ABSTRACT = re.compile(r"^\s*abstract\s*[\u2014\u2013\-:\.]\s*(\S.*)$", re.IGNORECASE)
_FIG_CAPTION = re.compile(r"^(Fig(?:ure)?\.?\s*\d+[a-zA-Z]?)\s*[:.\-\u2014\u2013]\s*(\S.{4,})$", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^(Table\s+(?:\d+|[IVX]+))\s*[:.\-\u2014\u2013]\s*(\S.{4,})$", re.IGNORECASE)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)")
_MATH_CHARS = "\u2211\u222b\u2202\u2264\u2265\u2248\u2260\u00b1\u2208\u2207\u221a\u221e"


def _safe_name(name: str) -> str:
    base = (name or "paper.pdf").replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^\w.\- ]", "_", base)
    return base[:120] or "paper.pdf"


# ------------------------------------------------------------------ validation
def validate_pdf(data, filename: str = "paper.pdf") -> None:
    """Raise PdfError with a clear message if the upload cannot be processed."""
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise PdfError("The uploaded file is empty.")
    limit = config.MAX_PDF_MB * 1024 * 1024
    if len(data) > limit:
        raise PdfError(f"The file is {len(data) / 1024 / 1024:.1f} MB; the limit is {config.MAX_PDF_MB} MB.")
    if filename and not filename.lower().endswith(".pdf"):
        raise PdfError("Please upload a PDF file (the file name should end in .pdf).")
    if b"%PDF-" not in bytes(data[:1024]):
        raise PdfError("This file does not look like a PDF.")


# ------------------------------------------------------------------ text cleaning
EDGE_TOP, EDGE_BOTTOM = 2, 1                     # header/footer candidates: first 2 and last 1 non-empty lines of a page


def _edge_indices(lines: list) -> set:
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    return set(nonempty[:EDGE_TOP]) | set(nonempty[-EDGE_BOTTOM:])


def _clean_page_texts(pages: list) -> list:
    """Normalise ligatures, drop control characters, page numbers and repeated headers/footers, fix hyphenation.

    Headers, footers and page numbers are only removed from the top and bottom lines of each page, so body text
    that happens to repeat (common in narrow columns) is never lost."""
    cleaned = []
    for text in pages:
        for src, dst in _LIGATURES.items():
            text = text.replace(src, dst)
        cleaned.append(_CONTROL.sub(" ", text))

    def key(line):
        return re.sub(r"\d+", "#", line.strip().lower())
    counts = Counter()
    for text in cleaned:
        lines = text.splitlines()
        counts.update({key(lines[i]) for i in _edge_indices(lines) if 0 < len(lines[i].strip()) < 100})
    threshold = max(3, int(0.4 * len(cleaned)))
    repeated = {k for k, n in counts.items() if n >= threshold and len(cleaned) >= 3}

    out = []
    for text in cleaned:
        raw_lines = text.splitlines()
        edges = _edge_indices(raw_lines)
        lines = []
        for i, line in enumerate(raw_lines):
            stripped = line.strip()
            if i in edges and stripped:
                if re.fullmatch(r"(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?", stripped, re.IGNORECASE):
                    continue                                 # bare page numbers
                if key(line) in repeated:
                    continue                                 # running headers and footers
            lines.append(line.rstrip())
        joined = "\n".join(lines)
        out.append(re.sub(r"(\w)-\n([a-z])", r"\1\2", joined))   # re-join words hyphenated across lines
    return out


# ------------------------------------------------------------------ structure detection
def _detect_sections(lines: list) -> tuple:
    """Split lines into canonical sections. Returns ({name: reflowed text}, [names in order], front_text)."""
    return _split_sections(lines)[:3]


def _split_sections(lines: list) -> tuple:
    """Like _detect_sections, plus {name: raw lines} (needed for equation detection, which works line by line)."""
    buckets, order, front = {}, [], []
    current = None
    for line in lines:
        if len(line.strip()) <= 70:
            hit = next((name for name, rx in _HEADING_RES.items() if rx.match(line)), None)
            if hit:
                current = hit
                if hit not in order:
                    order.append(hit)
                buckets.setdefault(hit, [])
                continue
        inline = _INLINE_ABSTRACT.match(line)
        if inline and "abstract" not in order:
            current = "abstract"
            order.append("abstract")
            buckets["abstract"] = [inline.group(1)]
            continue
        (buckets[current] if current else front).append(line)
    sections = {k: " ".join(" ".join(v).split())[:MAX_SECTION_CHARS] for k, v in buckets.items()}
    return sections, order, "\n".join(front), buckets


def _is_equation_line(line: str) -> bool:
    line = line.strip()
    if not 6 <= len(line) <= 160:
        return False
    if not ("=" in line or any(ch in line for ch in _MATH_CHARS)):
        return False
    if len(re.findall(r"[A-Za-z]{5,}", line)) > 3:           # prose, not a formula
        return False
    return not re.search(r"\b(table|figure|fig\.)\b", line, re.IGNORECASE)


_TITLE_WORDS = {"for", "of", "the", "with", "in", "on", "to", "a", "an", "using", "via", "towards", "toward", "by",
                "and", "from", "at", "is", "are"}
_AFFILIATION = re.compile(r"@|university|institute|department|school of|laborator|college|corresponding|"
                          r"arxiv|http|vol\.|issn|doi:", re.IGNORECASE)


def _looks_like_authors(line: str) -> bool:
    """'A. Researcher, B. Scientist' / 'Jane Doe and John Roe': capitalised words, separators, no title-style words."""
    s = line.strip()
    words = s.replace(",", " ").replace("&", " ").split()
    if not 2 <= len(words) <= 14:
        return False
    if any(w.lower() in _TITLE_WORDS - {"and"} for w in words):
        return False
    capitalised = all(w[0].isupper() or w.lower() == "and" for w in words if w[0].isalpha())
    initials = sum(1 for w in words if re.fullmatch(r"[A-Z]\.?", w) or re.fullmatch(r"(?:[A-Z]\.){2,}", w))
    return capitalised and ("," in s or " and " in s or initials >= 1) and "-" not in s.replace("- ", "")


def _guess_title(front: str, metadata_title: Optional[str]) -> str:
    meta = (metadata_title or "").strip()
    if 10 <= len(meta) <= 250 and not re.search(r"untitled|microsoft word|\.(docx?|tex|pdf)$", meta, re.IGNORECASE):
        return " ".join(meta.split())
    title_lines = []
    for line in front.splitlines():
        s = line.strip()
        if not s:
            if title_lines:
                break
            continue
        if _AFFILIATION.search(s) or re.search(r"\babstract\b", s, re.IGNORECASE) or (title_lines and _looks_like_authors(s)):
            if title_lines:
                break
            continue
        if len(s) < 4 or re.fullmatch(r"[\d\W]+", s):
            continue
        title_lines.append(s)
        if len(title_lines) == 3:
            break
    return " ".join(title_lines)[:250]


# ------------------------------------------------------------------ tables
def _clean_cell(value) -> str:
    return " ".join(str(value or "").split())[:80]


def _plumber_tables(data: bytes, page_texts: list, deadline: float):
    """Tables found by pdfplumber, or None if pdfplumber is not installed."""
    try:
        import pdfplumber
    except ImportError:
        return None
    tables = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for number, page in enumerate(pdf.pages[:len(page_texts)], 1):
                if time.monotonic() > deadline:
                    break
                try:
                    found = page.extract_tables()
                except Exception:
                    continue
                captions = [m.group(0) for l in page_texts[number - 1].splitlines()
                            if (m := _TABLE_CAPTION.match(l.strip()))]
                for index, raw in enumerate(found):
                    rows = [[_clean_cell(c) for c in row[:10]] for row in raw[:25] if row]
                    filled = sum(1 for r in rows for c in r if c)
                    if len(rows) >= 2 and max(len(r) for r in rows) >= 2 and filled >= 0.5 * sum(len(r) for r in rows):
                        caption = captions[index] if len(captions) == len(found) else ""
                        tables.append({"page": number, "caption": caption, "rows": rows, "source": "pdfplumber"})
    except Exception as exc:
        log.warning("pdfplumber table extraction failed: %s", type(exc).__name__)
        return None                     # treated as 'unavailable': the caller falls back to the layout heuristic
    return tables


def _layout_tables(layout_pages: list) -> list:
    """Fallback: runs of lines that split into 3+ columns on wide gaps and contain numbers.

    pypdf's layout mode inserts blank lines between rows, so blank lines are ignored; a text-only line with 3+
    columns directly above the run is taken as the header row."""
    tables = []
    for number, text in enumerate(layout_pages, 1):
        run, numeric_rows, header = [], 0, None
        for line in [l for l in text.splitlines() if l.strip()] + [None]:
            cells = [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()] if line is not None else []
            numeric = any(re.search(r"\d", c) for c in cells[1:])
            if len(cells) >= 3 and numeric:
                if not run and header:
                    run.append(header)
                run.append(cells[:10])
                numeric_rows += 1
                continue
            if numeric_rows >= 3:
                tables.append({"page": number, "caption": "", "rows": run[:25], "source": "layout"})
            run, numeric_rows = [], 0
            header = cells[:10] if len(cells) >= 3 and not numeric else None
    return tables


# ------------------------------------------------------------------ main entry point
def extract_pdf(data: bytes, filename: str = "paper.pdf") -> dict:
    """Validate and read a PDF. Returns a dict (see module docstring); raises PdfError with a readable message."""
    validate_pdf(data, filename)
    started = time.monotonic()
    deadline = started + config.MAX_PDF_SECONDS
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(bytes(data)), strict=False)
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise PdfError("This PDF is password protected. Remove the password and upload it again.")
        total_pages = len(reader.pages)
    except PdfError:
        raise
    except Exception as exc:
        log.warning("pdf open failed: %s", type(exc).__name__)
        raise PdfError("This PDF could not be read; it may be corrupted. Try re-saving or re-exporting it.") from exc
    if total_pages == 0:
        raise PdfError("This PDF has no pages.")

    warnings = []
    pages_to_read = min(total_pages, config.MAX_PDF_PAGES)
    if total_pages > pages_to_read:
        warnings.append(f"Only the first {pages_to_read} of {total_pages} pages were analysed (page limit).")

    raw_pages, failed, timed_out = [], 0, False
    for index in range(pages_to_read):
        if time.monotonic() > deadline:
            warnings.append(f"Reading stopped after {index} pages because the time limit was reached.")
            timed_out = True
            break
        try:
            raw_pages.append(reader.pages[index].extract_text() or "")
        except Exception:
            raw_pages.append("")
            failed += 1
    if failed:
        warnings.append(f"{failed} page(s) could not be read and were skipped.")

    if timed_out and not raw_pages:
        raise PdfError("Reading this PDF took too long. Try a smaller file or fewer pages.")
    pages = _clean_page_texts(raw_pages)
    chars = sum(len(re.sub(r"\s", "", p)) for p in pages)
    if not pages or chars / max(len(pages), 1) < config.MIN_CHARS_PER_PAGE or chars < 500:
        raise PdfError("This PDF seems to contain scanned images rather than text. OCR is not supported, so please "
                       "upload a text-based PDF (for example, exported from Word, LaTeX or the publisher).")

    lines = [l for p in pages for l in p.splitlines()]
    sections, order, front, section_lines = _split_sections(lines)
    if len(order) < 2:
        warnings.append("Section headings could not be detected reliably; the opening and closing text are used instead.")
    abstract = sections.get("abstract", "")
    if not abstract:
        title_text = _guess_title(front, None)
        opening = [l.strip() for l in (front or "\n".join(lines[:40])).splitlines()
                   if l.strip() and l.strip() not in title_text and not _AFFILIATION.search(l)
                   and not _looks_like_authors(l)]
        abstract = re.sub(r"\s+", " ", " ".join(opening))[:1500].strip()
        warnings.append("No 'Abstract' heading was found; the opening text is used in its place.")

    body_names = [n for n in order if n not in ("references", "acknowledgements")]
    text_parts = [front] + [sections[n] for n in body_names if sections.get(n)]
    text = ("\n\n".join(p for p in text_parts if p.strip()) if len(order) >= 2 else "\n".join(lines))
    if len(text) > config.MAX_TEXT_CHARS:
        text = text[:config.MAX_TEXT_CHARS]
        warnings.append("The text was very long and was cut for analysis.")

    body_lines = [l for n in body_names for l in section_lines.get(n, [])] if len(order) >= 2 else lines
    equations, seen = [], set()
    for l in body_lines:
        s = " ".join(l.split())
        if _is_equation_line(s) and s not in seen:
            seen.add(s)
            equations.append(s)
    figures, table_captions = [], []
    for number, page in enumerate(pages, 1):
        for l in page.splitlines():
            s = l.strip()
            if (m := _FIG_CAPTION.match(s)):
                figures.append({"label": " ".join(m.group(1).split()), "caption": m.group(2)[:200], "page": number})
            elif (m := _TABLE_CAPTION.match(s)):
                table_captions.append({"page": number, "label": " ".join(m.group(1).split()), "caption": m.group(2)[:200]})

    tables = _plumber_tables(bytes(data), pages, deadline)
    if tables is None:
        warnings.append("Table extraction used a text-layout heuristic (pdfplumber is unavailable or failed; "
                        "install or update pdfplumber for better results).")
        layout = []
        for index in range(len(pages)):
            try:
                layout.append(reader.pages[index].extract_text(extraction_mode="layout") or "")
            except Exception:
                layout.append("")
        tables = _layout_tables(layout)
    covered = {(t["page"], t["caption"][:20]) for t in tables if t["caption"]}
    for tc in table_captions:                      # captions with no extracted grid are still listed
        if not any(t["page"] == tc["page"] and tc["label"] in t["caption"] for t in tables):
            tables.append({"page": tc["page"], "caption": f"{tc['label']}: {tc['caption']}", "rows": [], "source": "caption"})
    if not tables:
        warnings.append("No tables were detected.")
    warnings.append("Equations are recovered as text and may lose formatting; figures are listed by caption only "
                    "(data inside graphs is not read).")

    metadata_title = None
    try:
        metadata_title = (reader.metadata or {}).get("/Title")
    except Exception:
        pass
    first_page = pages[0] if pages else ""
    doi = _DOI.search(first_page)
    refs = "\n".join(section_lines.get("references", []))
    return {
        "filename": _safe_name(filename), "n_pages": total_pages, "pages_read": len(pages), "n_chars": len(text),
        "title_guess": _guess_title(front or first_page, metadata_title), "doi": doi.group(1).rstrip(".,;)") if doi else None,
        "abstract": " ".join(abstract.split()), "sections": {k: v for k, v in sections.items() if k != "references"},
        "section_order": order, "tables": tables[:30], "equations": equations[:20], "figures": figures[:25],
        "n_references": len(re.findall(r"^\s*(?:\[\d+\]|\d+\.)\s", refs, re.MULTILINE)),
        "text": text, "warnings": warnings,
        "seconds": round(time.monotonic() - started, 2),
    }
