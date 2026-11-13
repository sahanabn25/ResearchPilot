"""Text clean-up shared by the renderers."""
import re

PROVENANCE_LABELS = {"verified_from_papers": "Verified from papers", "user_provided": "User-provided",
                     "ai_generated": "AI-generated"}
PROVENANCE_COLORS = {"verified_from_papers": "2E8B57", "user_provided": "3A6EA5", "ai_generated": "C9731A"}

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_ASCII_MAP = {"\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2026": "...",
              "\u2265": ">=", "\u2264": "<=", "\u2192": "->", "\u2190": "<-", "\u00b1": "+/-", "\u2212": "-",
              "\u2022": "-", "\u00a0": " ", "\u2248": "~", "\u00d7": "x", "\u03bc": "u", "\u2713": "yes", "\u2717": "no"}


def clean(text) -> str:
    """Remove characters that are illegal in XML/DOCX and normalise whitespace (keeps newlines out of cells)."""
    return _CONTROL.sub("", "" if text is None else str(text))


def one_line(text, limit: int = 0) -> str:
    out = " ".join(clean(text).split())
    if limit and len(out) > limit:
        out = out[:limit - 1].rstrip() + "\u2026"
    return out


def latin1_safe(text: str) -> str:
    """For PDF fonts without Unicode coverage: map common symbols, drop what cp1252 cannot show."""
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text.encode("cp1252", "replace").decode("cp1252")
