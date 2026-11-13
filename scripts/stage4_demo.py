"""Live check of Stage 4: questionnaire + evaluation (uses your configured LLM; ~3 LLM calls after the topic run).

Examples
  # full run, answering the questions in the terminal
  python scripts/stage4_demo.py "machine learning for weather forecasting"
  # reuse a saved Stage 3 run (skips discovery/analysis) and fictional sample answers
  python scripts/stage4_demo.py --state outputs/stage3_run.json --answers data/sample_answers_weather.json
Answers files are JSON keyed by question id (q_title, q_problem, ...). Keys starting with "_" are ignored.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchpilot import config, llm  # noqa: E402
from researchpilot.memory import LongTermMemory, ShortTermMemory  # noqa: E402
from researchpilot.orchestrator import MissingAnswersError, Orchestrator  # noqa: E402
from researchpilot.state import WorkflowState  # noqa: E402


def ask(questions):
    print("\nAnswer about YOUR research project (press Enter to skip optional questions).\n")
    answers = {}
    for i, q in enumerate(questions, 1):
        tag = " (required)" if q["required"] else ""
        if q["kind"] == "gap":
            print(f"   Literature gap: {q['gap']}")
        answers[q["id"]] = input(f"{i}. {q['question']}{tag}\n   > ")
    return answers


def show(evaluation):
    print("\n=== Evaluation ===")
    if evaluation.get("message"):
        print("NOTE:", evaluation["message"])
    ev = evaluation["evidence"]
    print("\n[Checked by code] claimed improvements:")
    for c in ev["improvement_checks"]:
        print(f"  - {c['claim']}: {c['check']['status']} {c['check'].get('reason', '')}")
    print("[Checked by code] same-dataset comparisons with retrieved papers:")
    for r in ev["numeric_comparisons"]["comparisons"] or []:
        print(f"  - {r['metric']}: yours {r['user_value']} vs {r['paper_value']} ({r['title'][:50]}), better={r['user_value_better']}")
    if not ev["numeric_comparisons"]["comparisons"]:
        print("  (none comparable)")
    print("[Mentioned in your answers]", {k: v for k, v in ev["rigor"].items()})
    a = evaluation.get("assessment") or {}
    for g in a.get("gap_alignment", []):
        print(f"\n[AI] gap '{g['gap'][:70]}': {g['addressed']} - {g['rationale']}")
    if a:
        md = a["methodology_difference"]
        print(f"\n[AI] methodology vs existing work: {md['assessment']} - {md['rationale']}")
        for key, label in (("remaining_limitations", "limitation"), ("suggested_experiments", "experiment"),
                           ("potential_contributions", "contribution")):
            for item in a[key]:
                extra = f"  CAVEAT: {item['caveat']}" if item.get("caveat") else ""
                print(f"[{item['provenance']}] {label}: {item[label]}{extra}")
    print(f"\nDropped as unsupported: {evaluation['dropped_unsupported']}  | scope: {evaluation['scope_note']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="*")
    parser.add_argument("--state", help="saved WorkflowState JSON (from stage3_demo)")
    parser.add_argument("--answers", help="JSON file of answers keyed by question id")
    args = parser.parse_args()
    if not llm.is_configured():
        print(llm.missing_credentials_message())
        return

    orchestrator = Orchestrator(long_term=LongTermMemory(), short_term=ShortTermMemory(),
                                on_progress=lambda s, m, f: print(f"  [{f:4.0%}] {s}: {m}"))
    if args.state:
        with open(args.state, encoding="utf-8") as fh:
            state = WorkflowState.from_dict(json.load(fh))
        print(f"Loaded saved run for topic: {state.topic}")
    else:
        topic = " ".join(args.topic).strip() or "machine learning for weather forecasting"
        state = orchestrator.run_topic_analysis(topic)
    for err in state.errors:
        print("ERROR:", err["message"])
    if not state.analyses:
        return

    questions = state.questions or orchestrator.prepare_questions(state)
    if args.answers:
        with open(args.answers, encoding="utf-8") as fh:
            answers = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    else:
        answers = ask(questions)
    try:
        orchestrator.evaluate_user_work(state, answers)
    except MissingAnswersError as exc:
        print(exc)
        return
    for err in state.errors:
        print("ERROR:", err["message"])
    if state.evaluation:
        show(state.evaluation)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stage4_run.json", "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)
    print("\nFull run saved to outputs/stage4_run.json")


if __name__ == "__main__":
    main()
