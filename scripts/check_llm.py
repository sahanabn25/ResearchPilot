"""Check your LLM setup and show the REAL error if calls fail.   Usage: python scripts/check_llm.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from researchpilot import config, llm  # noqa: E402


def main():
    print("Provider      :", config.LLM_PROVIDER)
    print("Models (tried in order):", llm._model_chain())
    print("Configured    :", llm.is_configured())
    if not llm.is_configured():
        print(llm.missing_credentials_message())
        return
    try:
        print("\nModels your key can use:", ", ".join(llm.list_models()) or "(none listed)")
    except Exception as exc:
        print("\nCould not list models:", type(exc).__name__, llm.safe_message(exc, 300))
    print()
    for model in llm._model_chain():
        try:
            text, usage = llm._call_model('Return exactly this JSON: {"ok": true}', "Return JSON only.", 0.0, model)
            print(f"OK    {model}: {(text or '').strip()[:60]!r}  tokens={usage}")
        except Exception as exc:  # show everything useful, never the key
            print(f"FAIL  {model}: {type(exc).__name__} code={getattr(exc, 'code', None)}: {llm.safe_message(exc, 500)}")
    print("\nIf a model FAILs with 'not found', pick a name from the list above and set LLM_MODEL in .env.")


if __name__ == "__main__":
    main()
