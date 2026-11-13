"""Extra Stage 6 tests: strict prompt budgets, damaged PDFs, logging noise. Run with the whole suite."""
import logging
import os
import random
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_fixtures import make_paper_pdf, make_text_pdf  # noqa: E402

from researchpilot import config  # noqa: E402
from researchpilot.agents import paper_parser  # noqa: E402
from researchpilot.reporting import build_report, to_markdown  # noqa: E402
from researchpilot.state import WorkflowState  # noqa: E402
from researchpilot.reporting.render_md import escape_markdown  # noqa: E402
from researchpilot.tools import pdf_reader  # noqa: E402
from researchpilot.tools.pdf_reader import PdfError, extract_pdf  # noqa: E402

GOOD = make_paper_pdf()


def huge_extraction(sentence_words=20000):
    blob = ("word " * sentence_words).strip()
    return {"title_guess": "A very long paper", "abstract": "a " * 3000, "filename": "p.pdf", "text": blob,
            "sections": {name: blob for name in ("introduction", "methodology", "results", "conclusion",
                                                  "limitations", "future_work", "discussion")},
            "tables": [{"page": 3, "caption": f"Table {i}: caption", "rows": [["a", "b", "c"]] * 8, "source": "x"}
                       for i in range(40)]}


class TestPromptBudgetsAreStrict(unittest.TestCase):
    def test_long_papers_are_cut_to_the_budget(self):
        budget = 4000
        ex = huge_extraction()
        with mock.patch.object(config, "PARSE_CHAR_BUDGET", budget):
            overview, cut_a = paper_parser.build_overview_prompt(ex)
            results, cut_b = paper_parser.build_results_prompt(ex)
        # overview: abstract (<= 2000) + headers + intro (<= 2000) + method (budget-bounded, min 1500)
        self.assertLess(len(overview), budget + 2000 + 1000, len(overview))
        self.assertLess(len(results), budget * 1.15 + 600, len(results))
        self.assertIn("methodology", cut_a)
        self.assertIn("introduction", cut_a)
        self.assertTrue({"results", "conclusion"} <= set(cut_b))

    def test_a_huge_table_collection_is_capped(self):
        with mock.patch.object(config, "PARSE_CHAR_BUDGET", 3000):
            prompt, _ = paper_parser.build_results_prompt(huge_extraction(50))
        self.assertLess(len(prompt), 3000 * 1.15 + 600)

    def test_default_budget_keeps_a_call_within_free_tier_token_limits(self):
        overview, _ = paper_parser.build_overview_prompt(huge_extraction())
        results, _ = paper_parser.build_results_prompt(huge_extraction())
        for prompt in (overview, results):
            self.assertLess(len(prompt) / 4 + 400, 6000)            # ~tokens incl. instructions; free tiers allow ~6-8k/min


class TestDamagedFilesNeverCrash(unittest.TestCase):
    def assert_clean(self, data):
        try:
            ex = extract_pdf(data, "p.pdf")
        except PdfError as exc:
            self.assertTrue(str(exc))                                # readable message
        else:
            self.assertIn("text", ex)

    def test_truncated_garbage_and_junk_prefixed_files(self):
        rng = random.Random(3)
        cases = [GOOD[: len(GOOD) // 2], GOOD[: int(len(GOOD) * 0.9)], b"%PDF-1.7\n" + bytes(rng.randrange(256) for _ in range(4000)),
                 b"%PDF-1.4\n", b"junk before the header\r\n" + GOOD]
        for data in cases:
            self.assert_clean(data)

    def test_random_bit_flips(self):
        for seed in range(8):
            rng, data = random.Random(seed), bytearray(GOOD)
            for _ in range(150):
                data[rng.randrange(len(data))] = rng.randrange(256)
            self.assert_clean(bytes(data))

    def test_short_but_real_text_document_is_accepted(self):
        pdf = make_text_pdf("A short note about forecasting with neural networks on regional data sets. " * 12)
        self.assertGreater(extract_pdf(pdf, "note.pdf")["n_chars"], 500)

    def test_input_bytes_are_not_modified(self):
        data = bytearray(GOOD)
        extract_pdf(bytes(data), "p.pdf")
        self.assertEqual(bytes(data), GOOD)

    def test_noisy_pdf_library_logging_is_silenced(self):
        for name in ("pypdf", "pdfminer"):
            self.assertGreaterEqual(logging.getLogger(name).level, logging.ERROR)


class TestReaderOptions(unittest.TestCase):
    def test_pdfplumber_missing_degrades_with_a_warning(self):
        with mock.patch.object(pdf_reader, "_plumber_tables", return_value=None):
            ex = extract_pdf(GOOD, "p.pdf")
        self.assertTrue(any("pdfplumber" in w for w in ex["warnings"]))
        self.assertTrue(ex["tables"])

    def test_pdfplumber_crash_falls_back_to_the_layout_heuristic(self):
        with mock.patch("pdfplumber.open", side_effect=RuntimeError("boom")):
            ex = extract_pdf(GOOD, "p.pdf")
        grids = [t for t in ex["tables"] if t["rows"]]
        self.assertTrue(grids and grids[0]["source"] == "layout")
        self.assertTrue(any("pdfplumber" in w for w in ex["warnings"]))


HOSTILE = "![x](https://evil.example/p.png) [click](javascript:alert(1)) <img src=x onerror=alert(1)> **bold** `code` | pipe"
UNESCAPED_IMAGE = re.compile(r"(?<!\\)!\[")
UNESCAPED_LINK = re.compile(r"(?<!\\)\]\(")


class TestUntrustedTextCannotRenderAsMarkdown(unittest.TestCase):
    """Text from PDFs and scholarly APIs is shown on screen; it must never become an image, link or HTML."""

    @classmethod
    def setUpClass(cls):
        from test_stage6 import PAPER, upload_orchestrator
        o, _ = upload_orchestrator()
        cls.state = o.run_paper_analysis(PAPER, "paper.pdf")
        cls.state.topic = "Topic " + HOSTILE
        cls.state.profile["fields"]["methodology"] = HOSTILE
        cls.state.profile["fields"]["title"] = "Title " + HOSTILE
        first = next(iter(cls.state.analyses))
        cls.state.analyses[first]["title"] = "Related " + HOSTILE
        cls.state.paper["figures"][0]["caption"] = HOSTILE
        paper = cls.state.papers[0]
        paper.update({"title": "Paper " + HOSTILE, "authors": [HOSTILE], "venue": HOSTILE, "abstract": HOSTILE})
        cls.report = build_report(cls.state)

    def test_preview_markdown_is_escaped(self):
        md = to_markdown(self.report, safe=True)
        self.assertIsNone(UNESCAPED_IMAGE.search(md))
        self.assertIsNone(UNESCAPED_LINK.search(md))
        self.assertIsNone(re.search(r"(?<!\\)<img", md))              # an escaped \\<img is only text
        self.assertNotIn("**bold**", md)
        self.assertIn("\\!\\[x\\]", md)                       # still readable, just literal

    def test_downloadable_markdown_keeps_the_original_text(self):
        md = to_markdown(self.report)
        self.assertIn("![x](https://evil.example/p.png)", md)

    def test_tables_stay_well_formed_in_safe_mode(self):
        lines = [l for l in to_markdown(self.report, safe=True).splitlines() if l.startswith("| ")]
        self.assertTrue(lines)
        by_block, current = [], []
        for l in lines:
            current.append(len(re.findall(r"(?<!\\)\|", l)))
        self.assertTrue(all(n >= 3 for n in current))

    def test_escape_markdown_unit(self):
        self.assertEqual(escape_markdown("a*b_c[d](e)!"), "a\\*b\\_c\\[d\\]\\(e\\)\\!")
        self.assertEqual(escape_markdown(None), "")

    def test_uploaded_paper_tab_escapes_pdf_text(self):
        from test_ui_smoke import make_stub, run_app
        st = make_stub()
        st.session_state["wf"] = self.state
        run_app(st)
        shown = [str(c.args[0]) for fn in (st.markdown, st.caption, st.warning, st.expander, st.write)
                 for c in fn.call_args_list if c.args]
        self.assertTrue(any("evil.example" in text for text in shown))     # the text is shown...
        for text in shown:
            self.assertIsNone(UNESCAPED_IMAGE.search(text), text[:120])    # ...but never as an image or link
            self.assertIsNone(UNESCAPED_LINK.search(text), text[:120])

    def test_report_preview_in_the_app_is_escaped(self):
        from test_ui_smoke import make_stub, run_app
        state = WorkflowState.from_dict(self.state.to_dict())
        state.report = build_report(state)
        st = make_stub()
        st.session_state["wf"] = state
        run_app(st)
        previews = [str(c.args[0]) for c in st.markdown.call_args_list if c.args and "## 14. References" in str(c.args[0])]
        self.assertEqual(len(previews), 1)
        self.assertIn("evil.example", previews[0])
        self.assertIsNone(UNESCAPED_IMAGE.search(previews[0]))
        self.assertIsNone(UNESCAPED_LINK.search(previews[0]))


if __name__ == "__main__":
    unittest.main()
