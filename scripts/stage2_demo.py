"""Live check of Stage 2 (no LLM, no cost): discovery -> index -> retrieve -> memory.

Usage: python scripts/stage2_demo.py "machine learning for weather forecasting"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchpilot import retrieval  # noqa: E402
from researchpilot.discovery import DiscoveryError, discover_papers  # noqa: E402
from researchpilot.memory import LongTermMemory  # noqa: E402


def main():
    topic = " ".join(sys.argv[1:]).strip() or "machine learning for weather forecasting"
    memory = LongTermMemory()
    try:
        result = discover_papers(topic, limit=10, memory=memory)
    except DiscoveryError as exc:
        print("All sources failed:", exc)
        return
    print(f"\nTopic: {topic}   (from cache: {result['from_cache']})")
    print("Source status:", result["source_status"])
    papers = result["papers"]
    with_abs = sum(1 for p in papers if p["abstract"])
    print(f"{len(papers)} unique papers, {with_abs} with abstracts\n")
    for i, p in enumerate(papers, 1):
        print(f"{i:2d}. [{'+'.join(p['sources'])}] {p['title']} ({p['year']})  abstract={'yes' if p['abstract'] else 'NO'}")

    store = retrieval.get_store()
    n = retrieval.index_papers(store, papers)
    print(f"\nVector store: {store.backend_name}, {n} chunks indexed")
    for q in ("what datasets were used", "limitations of the method"):
        print(f"\nQuery: {q}")
        for h in retrieval.retrieve(store, q, k=3):
            print(f"  {h.score:.3f}  {h.metadata['title'][:70]}")
    memory.save_run("topic", topic, {"papers": len(papers), "with_abstract": with_abs})
    print("\nSaved run. Run this again: discovery should now come from cache (no API calls).")


if __name__ == "__main__":
    main()
