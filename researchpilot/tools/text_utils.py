"""Small text-cleaning helpers shared by tools."""
import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(value) -> str:
    """Strip markup (e.g. Crossref JATS tags), decode entities, collapse whitespace."""
    if value is None:
        return ""
    text = _TAG_RE.sub(" ", str(value))
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def clean_abstract(value) -> str:
    text = clean_text(value)
    return re.sub(r"^abstract[:.\s]+", "", text, flags=re.IGNORECASE).strip()
