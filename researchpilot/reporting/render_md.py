"""Markdown rendering of the report: used for the in-app preview and as a plain-text download.

`to_markdown(report)` is for files (clean text). `to_markdown(report, safe=True)` is for on-screen preview: all
report content is escaped so text taken from PDFs or scholarly APIs cannot render as images, links or HTML
(an embedded image link would make the browser contact an outside server when the preview is shown).
"""
from .text import PROVENANCE_LABELS, clean

_MD_SPECIAL = "\\`*_[]()<>!#|~"


def escape_markdown(text) -> str:
    """Make untrusted text render literally in Markdown (no links, images, HTML or emphasis from the text itself)."""
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in clean(text))


def _tag(prov, labels=None) -> str:
    labels = labels or PROVENANCE_LABELS
    return f"**[{labels[prov]}]** " if prov in labels else ""


def to_markdown(report: dict, safe: bool = False) -> str:
    labels = report.get("labels") or PROVENANCE_LABELS
    t = escape_markdown if safe else clean               # applied to every piece of CONTENT, never to our own markup

    def cell(text) -> str:
        return t(text).replace("|", "\\|").replace("\n", " ") if not safe else t(text).replace("\n", " ")

    def table(block) -> str:
        head = "| " + " | ".join(cell(h) for h in block["header"]) + " |"
        sep = "|" + "|".join("---" for _ in block["header"]) + "|"
        rows = ["| " + " | ".join(cell(c) for c in r) + " |" for r in block["rows"]]
        out = [head, sep] + rows
        if block.get("caption"):
            out.append(f"\n*{t(block['caption'])}*")
        return "\n".join(out)

    out = [f"# {t(report['title'])}", f"*{t(report['subtitle'])}*", ""]
    for notice in report.get("notices", []):
        out.append(f"> **Notice:** {t(notice)}")
    if report.get("notices"):
        out.append("")
    for section in report["sections"]:
        out += [f"## {section['number']}. {t(section['title'])}", ""]
        for b in section["blocks"]:
            kind = b["type"]
            if kind == "paragraph":
                out.append(_tag(b.get("provenance"), labels) + t(b["text"]))
            elif kind == "subheading":
                out.append(f"### {t(b['text'])}")
            elif kind == "note":
                out.append(f"*{t(b['text'])}*")
            elif kind == "bullets":
                if _tag(b.get("provenance"), labels):
                    out += [_tag(b["provenance"], labels).strip(), ""]       # blank line so the list renders as a list
                out.extend(f"- {t(i)}" for i in b["items"])
            elif kind == "references":
                out.extend(t(i) + "\n" for i in b["items"])
            elif kind == "table":
                if _tag(b.get("provenance"), labels):
                    out += [_tag(b["provenance"], labels).strip(), ""]       # blank line so the table renders as a table
                out.append(table(b))
            out.append("")
    return "\n".join(out).strip() + "\n"
