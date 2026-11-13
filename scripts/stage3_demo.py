"""Live check of Stage 3 (uses your configured LLM; roughly 3-5 LLM calls per run).

Usage: python scripts/stage3_demo.py "machine learning for weather forecasting"
Writes the full run (state, trace, analyses, gaps) to outputs/stage3_run.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchpilot import config, llm  # noqa: E402
from researchpilot.memory import LongTermMemory, ShortTermMemory  # noqa: E402
from researchpilot.orchestrator import Orchestrator  # noqa: E402


def main():
    topic = " ".join(sys.argv[1:]).strip() or "machine learning for weather forecasting"
    if not llm.is_configured():
        print(llm.missing_credentials_message())
        return

    def progress(stage, message, fraction):
        print(f"  [{fraction:4.0%}] {stage}: {message}")

    orchestrator = Orchestrator(long_term=LongTermMemory(), short_term=ShortTermMemory(), on_progress=progress)
    print(f"Topic: {topic}\n")
    state = orchestrator.run_topic_analysis(topic)

    print(f"\nSteps used: {state.steps_used}/{orchestrator.max_steps}   Stages: {state.completed_stages}")
    print(f"Source status: {state.source_status}")
    print(f"Analysis stats: {state.analysis_stats}")
    for err in state.errors:
        print("ERROR:", err["message"])

    print("\n--- Paper analyses ---")
    for a in state.analyses.values():
        print(f"\n[{a['status']}] {a['title'][:90]} ({a['year']})  evidence={a['evidence_level']}")
        if a["status"] == "ok":
            print(f"  problem   : {a['research_problem']}\n  method    : {a['methodology']}")
            print(f"  datasets  : {a['datasets']}  metrics: {a['metrics']}\n  results   : {a['key_results']}")
            if any(a["verification"]["dropped_items"].values()) or a["verification"]["unverified_numbers"]:
                print(f"  VERIFICATION removed: {a['verification']}")

    g = state.gaps
    if g:
        print("\n--- Research gaps ---")
        print(g.get("message") or g["scope_note"])
        for gap in g["gaps"]:
            print(f"\n* {gap['gap']}  [{gap['evidence_strength']}; {len(gap['paper_ids'])} paper(s)]\n  {gap['rationale']}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stage3_run.json", "w", encoding="utf-8") as fh:
        json.dump(state.to_dict(), fh, indent=2, ensure_ascii=False)
    print("\nFull trace saved to outputs/stage3_run.json")


if __name__ == "__main__":
    main()
