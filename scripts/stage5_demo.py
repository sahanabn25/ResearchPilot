"""Stage 5 demo: build the research analysis report (PDF, DOCX, Markdown).

Examples
  # OFFLINE backup: use a saved run, no LLM needed
  python scripts/stage5_demo.py --state outputs/stage4_run.json --no-llm
  # saved Stage 3 run + sample answers (uses your LLM for the questionnaire/evaluation, and for the narrative)
  python scripts/stage5_demo.py --state outputs/stage3_run.json --answers data/sample_answers_weather.json
  # everything from scratch
  python scripts/stage5_demo.py "machine learning for weather forecasting" --answers data/sample_answers_weather.json
Files are written to outputs/ (override with --out).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchpilot import llm  # noqa: E402
from researchpilot.memory import LongTermMemory, ShortTermMemory  # noqa: E402
from researchpilot.orchestrator import MissingAnswersError, Orchestrator  # noqa: E402
from researchpilot.reporting import to_docx_bytes, to_markdown, to_pdf_bytes  # noqa: E402
from researchpilot.state import WorkflowState  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="*")
    parser.add_argument("--state", help="saved WorkflowState JSON (outputs/stage3_run.json or stage4_run.json)")
    parser.add_argument("--answers", help="JSON file of answers keyed by question id")
    parser.add_argument("--no-llm", action="store_true", help="computed text only; no narrative LLM call")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    orchestrator = Orchestrator(long_term=LongTermMemory(), short_term=ShortTermMemory(),
                                on_progress=lambda s, m, f: print(f"  [{f:4.0%}] {s}: {m}"))
    if args.state:
        with open(args.state, encoding="utf-8") as fh:
            state = WorkflowState.from_dict(json.load(fh))
        print(f"Loaded saved run for topic: {state.topic}")
    else:
        if not llm.is_configured():
            print(llm.missing_credentials_message())
            return
        state = orchestrator.run_topic_analysis(" ".join(args.topic).strip() or "machine learning for weather forecasting")

    if args.answers:
        if not llm.is_configured():
            print(llm.missing_credentials_message())
            return
        with open(args.answers, encoding="utf-8") as fh:
            answers = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
        if not state.questions:
            orchestrator.prepare_questions(state)
        try:
            orchestrator.evaluate_user_work(state, answers)
        except MissingAnswersError as exc:
            print(exc)
            return

    if not args.no_llm and not llm.is_configured():
        print("No LLM configured: building the report with computed text only.\n")
        args.no_llm = True
    report = orchestrator.generate_report(state, use_llm=not args.no_llm)
    for err in state.errors:
        print("ERROR:", err["message"])
    if not report:
        return

    os.makedirs(args.out, exist_ok=True)
    for name, data, mode in (("report.pdf", to_pdf_bytes(report), "wb"), ("report.docx", to_docx_bytes(report), "wb"),
                             ("report.md", to_markdown(report), "w")):
        with open(os.path.join(args.out, name), mode, **({} if "b" in mode else {"encoding": "utf-8"})) as fh:
            fh.write(data)
    with open(os.path.join(args.out, "report_run.json"), "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)
    print(f"\nReport: '{report['title']}'")
    print(f"Complete: {report['complete']}  | narrative: {report.get('narrative_source')}")
    for notice in report["notices"]:
        print("  NOTICE:", notice)
    print(f"Written to {args.out}/: report.pdf, report.docx, report.md, report_run.json")


if __name__ == "__main__":
    main()
