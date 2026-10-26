
import os
import sys
import json
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CROSSREF_URL = "https://api.crossref.org/works"

# Initialize Gemini only when the API key is available
client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


def search_papers(topic, limit=5):
    """Tool: retrieve scholarly metadata from Crossref."""

    response = requests.get(
        CROSSREF_URL,
        params={
            "query.bibliographic": topic,
            "rows": limit,
            "select": "DOI,title,author,published,URL",
        },
        timeout=15,
    )
    response.raise_for_status()

    items = response.json().get("message", {}).get("items", [])
    papers = []

    for item in items:
        titles = item.get("title", [])
        if not titles:
            continue

        date_info = (
            item.get("published-print")
            or item.get("published-online")
            or item.get("published")
            or {}
        )

        date_parts = date_info.get("date-parts", [[]])
        year = (
            date_parts[0][0]
            if date_parts and date_parts[0]
            else None
        )

        authors = []
        for author in item.get("author", []):
            name = " ".join(
                part
                for part in [
                    author.get("given"),
                    author.get("family"),
                ]
                if part
            )
            if name:
                authors.append(name)

        papers.append({
            "title": titles[0],
            "authors": authors,
            "year": year,
            "doi": item.get("DOI"),
            "url": item.get("URL"),
        })

    return papers


def create_research_plan(topic, papers):
    """Generate and validate a research plan using Gemini."""

    if client is None:
        raise ValueError(
            "GEMINI_API_KEY is missing from your .env file."
        )

    instructions = """
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

    prompt = (
        "Create a research plan in JSON format for this topic:\n"
        + topic
        + "\n\nRetrieved scholarly papers:\n"
        + json.dumps(papers, ensure_ascii=False)
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=instructions,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    result = json.loads(response.text)

    required_keys = {
        "topic",
        "research_questions",
        "summary",
        "relevant_paper_indices",
        "limitations",
    }

    if not isinstance(result, dict) or set(result) != required_keys:
        raise ValueError(
            "The model returned an unexpected JSON structure."
        )

    if not isinstance(result["research_questions"], list):
        raise ValueError("research_questions must be a list.")

    if not all(
        isinstance(question, str)
        for question in result["research_questions"]
    ):
        raise ValueError(
            "Every research question must be a string."
        )

    if not isinstance(result["relevant_paper_indices"], list):
        raise ValueError(
            "relevant_paper_indices must be a list."
        )

    if not all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 1 <= index <= len(papers)
        for index in result["relevant_paper_indices"]
    ):
        raise ValueError(
            "The model returned an invalid paper index."
        )

    if not isinstance(result["summary"], str):
        raise ValueError("summary must be a string.")

    if not isinstance(result["limitations"], str):
        raise ValueError("limitations must be a string.")

    result["papers"] = [
        papers[index - 1]
        for index in result["relevant_paper_indices"]
    ]

    return result



def main():
    topic = " ".join(sys.argv[1:]).strip()

    if not topic:
        topic = input("Enter a research topic: ").strip()

    if not topic:
        print("Error: Research topic cannot be empty.")
        return

    if not GEMINI_API_KEY:
        print("Error: Add GEMINI_API_KEY to your .env file.")
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
            print(f"\n{index}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors']) or 'Unknown'}")
            print(f"   Year: {paper['year'] or 'Unknown'}")
            print(f"   DOI: {paper['doi'] or 'Not available'}")
            print(f"   URL: {paper['url'] or 'Not available'}")

        print("\n" + "-" * 60)
        print("\nGenerating the research plan...")

        result = create_research_plan(topic, papers)

        print("\nResearchPilot output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except requests.RequestException as exc:
        print(f"Search service error: {exc}")

    except json.JSONDecodeError:
        print("Error: The model returned invalid JSON.")

    except ValueError as exc:
        print(f"Configuration or validation error: {exc}")

    except Exception as exc:
        print(f"Agent failed: {type(exc).__name__}")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
