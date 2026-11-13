"""Stage 6 demo (Option 2): analyse an uploaded research paper (PDF) and build the report.

Examples
  # OFFLINE check of how well your PDF is read (no LLM, no network): title, sections, tables, equations, figures
  python scripts/stage6_demo.py path/to/paper.pdf --extract-only
  # full run with your LLM (about 5 LLM calls; searches Crossref, OpenAlex and arXiv for related work)
  python scripts/stage6_demo.py path/to/paper.pdf
  # try it on the fictional sample
  python scripts/make_sample_paper.py && python scripts/stage6_demo.py data/sample_paper_FICTIONAL.pdf
Files are written to outputs/ (override with --out): paper_report.pdf, paper_report.docx, paper_report.md, paper_run.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchpilot import llm  # noqa: E402
from researchpilot.memory import LongTermMemory, ShortTermMemory  # noqa: E402
from researchpilot.orchestrator import Orchestrator  # noqa: E402
from researchpilot.reporting import to_docx_bytes, to_markdown, to_pdf_bytes  # noqa: E402
from researchpilot.tools.pdf_reader import PdfError, extract_pdf  # noqa: E402


def show_extraction(ex):
    print(f"\nFile: {ex['filename']}   pages read: {ex['pages_read']} of {ex['n_pages']}   text: {ex['n_chars']} characters"
          f"   ({ex['seconds']} s)")
    print(f"Title guess : {ex['title_guess']!r}")
    print(f"DOI         : {ex['doi']}")
    print(f"Sections    : {', '.join(ex['section_order']) or 'none detected'}   (references counted: {ex['n_references']})")
    print(f"Abstract    : {ex['abstract'][:300]}{'...' if len(ex['abstract']) > 300 else ''}")
    print(f"\nTables ({len(ex['tables'])}):")
    for t in ex["tables"]:
        size = f"{len(t['rows'])} rows" if t["rows"] else "caption only"
        print(f"  p.{t['page']}  [{t['source']}] {t['caption'][:70] or '(no caption)'}  -> {size}")
        for row in t["rows"][:3]:
            print("      ", " | ".join(row))
    print(f"\nEquation-like lines ({len(ex['equations'])}):")
    for eq in ex["equations"][:6]:
        print("  ", eq)
    print(f"\nFigure captions ({len(ex['figures'])}):")
    for f in ex["figures"][:6]:
        print(f"  p.{f['page']}  {f['label']}: {f['caption'][:70]}")
    print("\nWarnings:")
    for w in ex["warnings"]:
        print("  -", w)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--extract-only", action="store_true", help="offline: only read the PDF and show what was found")
    parser.add_argument("--no-llm-report", action="store_true", help="computed text only in the report narrative")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    with open(args.pdf, "rb") as fh:
        data = fh.read()
    name = os.path.basename(args.pdf)
    if args.extract_only:
        try:
            show_extraction(extract_pdf(data, name))
        except PdfError as exc:
            print("Cannot use this PDF:", exc)
        return
    if not llm.is_configured():
        print(llm.missing_credentials_message())
        return

    orchestrator = Orchestrator(long_term=LongTermMemory(), short_term=ShortTermMemory(),
                                on_progress=lambda s, m, f: print(f"  [{f:4.0%}] {s}: {m}"))
    state = orchestrator.run_paper_analysis(data, name)
    for err in state.errors:
        print("ERROR:", err["message"])
    if not state.paper:
        return
    show_extraction(state.paper)
    parsed = state.paper.get("parsed", {})
    print(f"\nAI extraction: {parsed.get('source')} ({parsed.get('llm_calls')} LLM calls)")
    for note in parsed.get("notes", []):
        print("  NOTE:", note)
    claims = parsed.get("claims", {})
    print("Techniques:", claims.get("techniques"), "| Datasets:", claims.get("datasets"))
    print("Metrics   :", [(m["name"], m["value"]) for m in claims.get("metrics", [])])
    print("Removed as unsupported by the paper's own text:", claims.get("dropped"))
    print("Related-work queries:", parsed.get("queries"))
    print(f"Related papers: {len(state.papers)}  analysed: {state.analysis_stats.get('analysed', 0)}  steps: {state.steps_used}")
    if state.gaps and state.gaps.get("gaps"):
        print("Gaps in the related literature:")
        for g in state.gaps["gaps"]:
            print(f"  - {g['gap']}  [{g['evidence_strength']}]")
    for g in (state.evaluation.get("assessment") or {}).get("gap_alignment", []):
        print(f"  paper vs gap '{g['gap'][:50]}': {g['addressed']}")

    report = orchestrator.generate_report(state, use_llm=not args.no_llm_report)
    if report:
        os.makedirs(args.out, exist_ok=True)
        for fname, payload, mode in (("paper_report.pdf", to_pdf_bytes(report), "wb"), ("paper_report.docx", to_docx_bytes(report), "wb"),
                                     ("paper_report.md", to_markdown(report), "w")):
            with open(os.path.join(args.out, fname), mode, **({} if "b" in mode else {"encoding": "utf-8"})) as fh:
                fh.write(payload)
        with open(os.path.join(args.out, "paper_run.json"), "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)
        print(f"\nReport '{report['title']}' written to {args.out}/ (paper_report.pdf, .docx, .md, paper_run.json)")
        for notice in report["notices"]:
            print("  NOTICE:", notice)


if __name__ == "__main__":
    main()
