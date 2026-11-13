"""ResearchPilot backend (Week 2 API preserved).

Public names kept for backward compatibility with app.py and the CLI:
GEMINI_API_KEY, MODEL, CROSSREF_URL, search_papers, create_research_plan, main.
Implementation now lives in the `researchpilot` package.
"""
import json
import sys

import requests

from researchpilot import config as _config
from researchpilot.config import CROSSREF_URL  # noqa: F401
from researchpilot.llm import LLMError, generate_json, is_configured, missing_credentials_message
from researchpilot.tools.crossref import search_papers  # noqa: F401

# Kept for the Week 2 app.py, which only tests GEMINI_API_KEY for truthiness: it is truthy whenever ANY
# LLM provider is configured (Groq is the default now). MODEL is the active provider's model.
GEMINI_API_KEY = ("configured" if is_configured() else None)
MODEL = _config.GEMINI_MODEL if _config.LLM_PROVIDER == "gemini" else _config.LLM_MODEL

REQUIRED_PLAN_KEYS = {
    "topic", "research_questions", "summary", "relevant_paper_indices", "limitations",
}

PLAN_INSTRUCTIONS = """
    You are ResearchPilot, an AI research assistant.

    Create a research plan using the supplied scholarly records.
    Return valid JSON only, with exactly these keys:
    topic, research_questions, summary,
    relevant_paper_indices, limitations.

    Requirements:
    - research_questions must be a list of strings.
    - relevant_paper_indices must be a list of integers.
    - Use 1-based indices corresponding to the supplied papers.
    - Select only relevant papers from the supplied records.
    - Do not invent paper titles, authors, or publication details.
    - Mention uncertainty and evidence limitations where appropriate.
    - Do not include Markdown fences or text outside the JSON object.
    """


def validate_plan(result, papers):
    """Validate the model's plan (same rules as Week 2) and attach selected papers."""
    if not isinstance(result, dict) or set(result) != REQUIRED_PLAN_KEYS:
        raise ValueError("The model returned an unexpected JSON structure.")
    if not isinstance(result["research_questions"], list):
        raise ValueError("research_questions must be a list.")
    if not all(isinstance(q, str) for q in result["research_questions"]):
        raise ValueError("Every research question must be a string.")
    if not isinstance(result["relevant_paper_indices"], list):
        raise ValueError("relevant_paper_indices must be a list.")
    if not all(
        isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= len(papers)
        for i in result["relevant_paper_indices"]
    ):
        raise ValueError("The model returned an invalid paper index.")
    if not isinstance(result["summary"], str):
        raise ValueError("summary must be a string.")
    if not isinstance(result["limitations"], str):
        raise ValueError("limitations must be a string.")

    result["papers"] = [papers[i - 1] for i in result["relevant_paper_indices"]]
    return result


def create_research_plan(topic, papers):
    """Generate and validate a research plan using Gemini."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Research topic cannot be empty.")
    if not papers:
        raise ValueError("No papers supplied for planning.")

    prompt = (
        "Create a research plan in JSON format for this topic:\n"
        + topic
        + "\n\nRetrieved scholarly papers:\n"
        + json.dumps(papers, ensure_ascii=False)
    )
    result = generate_json(prompt, system_instruction=PLAN_INSTRUCTIONS, temperature=0.2)
    return validate_plan(result, papers)


def _print_paper(index, paper):
    authors = ", ".join(paper["authors"]) or "Unknown"
    year = paper["year"] or "Unknown"
    doi = paper["doi"] or "Not available"
    url = paper["url"] or "Not available"
    print(f"\n{index}. {paper['title']}")
    print(f"   Authors: {authors}")
    print(f"   Year: {year}")
    print(f"   DOI: {doi}")
    print(f"   URL: {url}")


def main():
    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        topic = input("Enter a research topic: ").strip()
    if not topic:
        print("Error: Research topic cannot be empty.")
        return
    if not is_configured():
        print("Error: " + missing_credentials_message())
        return

    try:
        print("\nSearching scholarly records...")
        papers = search_papers(topic)
        if not papers:
            print("No paper records found. Try a more specific topic.")
            return

        print(f"\nRetrieved {len(papers)} paper records:")
        print("-" * 60)
        for index, paper in enumerate(papers, start=1):
            _print_paper(index, paper)
        print("\n" + "-" * 60)
        print("\nGenerating the research plan...")

        result = create_research_plan(topic, papers)
        print("\nResearchPilot output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except requests.RequestException as exc:
        print(f"Search service error: {exc}")
    except LLMError as exc:
        print(f"Language-model error: {exc}")
    except ValueError as exc:
        print(f"Configuration or validation error: {exc}")
    except Exception as exc:
        print(f"Agent failed: {type(exc).__name__}")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
