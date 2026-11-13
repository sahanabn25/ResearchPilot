"""Report Generation agent: assembles the final research analysis report.

Deterministic content comes from reporting.builder (no LLM). One optional LLM call adds validated narrative
paragraphs; if it fails or its output is rejected, the report is still produced with computed text, and a notice
says so. The result is a JSON-serialisable dict (see reporting/builder.py) that the renderers turn into a
preview, DOCX and PDF.
"""
from typing import Callable, Optional

from .. import config
from ..llm import generate_json
from ..logging_setup import get_logger
from ..reporting.builder import build_report, citation_problems, validate_report
from ..reporting.narrative import generate_narrative

log = get_logger(__name__)


def write_report(state, llm: Callable = generate_json, use_llm: Optional[bool] = None) -> dict:
    use_llm = config.REPORT_USE_LLM if use_llm is None else use_llm
    narrative, note = None, ""
    if use_llm:
        try:
            narrative = generate_narrative(state, llm=llm)
            if narrative is None:
                note = "AI narrative paragraphs were not usable, so computed summaries are shown instead."
        except Exception as exc:  # any LLM/validation problem: fall back, never fail the report
            note = "AI narrative paragraphs were unavailable, so computed summaries are shown instead."
            log.warning("report narrative failed: %s", type(exc).__name__)
    report = build_report(state, narrative)
    if note:
        report["notices"].append(note)
    validate_report(report)
    problems = citation_problems(report)
    if problems:  # user text can contain stray [n]; keep the report but record it
        log.warning("report citation check: %s", problems[:3])
    report["narrative_source"] = "llm" if narrative else "computed"
    return report
