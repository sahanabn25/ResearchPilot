"""Smoke test for app_workflow.py using a stub `streamlit` module and a scripted LLM.

This proves the script executes end-to-end over the real orchestrator code paths (no exceptions,
the right widgets/actions fire). It does NOT prove the visual layout; run the app once to check that.
"""
import os
import runpy
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from test_stage4 import ANSWERS, FlowLLM, fake_discover, flow_state  # noqa: E402
from test_stage5 import ReportLLM  # noqa: E402

from researchpilot import retrieval  # noqa: E402
from researchpilot.memory import LongTermMemory  # noqa: E402
from researchpilot.orchestrator import Orchestrator  # noqa: E402
from researchpilot.state import WorkflowState  # noqa: E402

APP = os.path.join(ROOT, "app_workflow.py")


def make_stub(form_clicks=(), text_value="", button_clicks=()):
    st = mock.MagicMock(name="streamlit")
    st.session_state = {}
    st.cache_resource = lambda f: f
    st.tabs.side_effect = lambda labels: [mock.MagicMock() for _ in labels]
    st.made_columns = []                                           # remembered so tests can inspect column widgets

    def columns(spec):
        cols = [mock.MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
        st.made_columns.extend(cols)
        return cols
    st.columns.side_effect = columns
    st.button.side_effect = lambda label, **kw: label in button_clicks
    st.checkbox.return_value = True
    st.form_submit_button.side_effect = lambda label, **kw: label in form_clicks
    st.text_input.return_value = text_value
    st.selectbox.return_value = 10
    return st


def run_app(st, **patches):
    tmp = tempfile.TemporaryDirectory()
    self_patches = [
        mock.patch.dict(sys.modules, {"streamlit": st}),
        mock.patch("researchpilot.memory.LongTermMemory", lambda *a, **k: LongTermMemory(os.path.join(tmp.name, "m.sqlite3"))),
        mock.patch("researchpilot.llm.is_configured", lambda: bool(patches.get("api_key", "test-key"))),
    ]
    if "orchestrator" in patches:
        self_patches.append(mock.patch("researchpilot.orchestrator.Orchestrator", patches["orchestrator"]))
    for p in self_patches:
        p.start()
    try:
        runpy.run_path(APP, run_name="__main__")
    finally:
        for p in reversed(self_patches):
            p.stop()
        tmp.cleanup()


def scripted_orchestrator(llm):
    return lambda **kw: Orchestrator(llm=llm, discover=fake_discover,
                                     store_factory=lambda: retrieval.LexicalStore(), **kw)


def button_labels(st):
    return [c.args[0] for c in st.button.call_args_list if c.args]


class TestAppWorkflowSmoke(unittest.TestCase):
    def test_empty_state_renders_home(self):
        st = make_stub()
        run_app(st)
        st.stop.assert_not_called()
        st.error.assert_not_called()
        self.assertEqual(st.session_state.get("wf"), None)

    def test_missing_api_key_stops_with_message(self):
        st = make_stub()
        st.stop.side_effect = SystemExit
        with self.assertRaises(SystemExit):
            run_app(st, api_key=None)
        self.assertIn("API_KEY", st.error.call_args.args[0])          # provider-specific key name, e.g. GROQ_API_KEY

    def test_blank_topic_is_rejected(self):
        st = make_stub(form_clicks=("Discover, analyse and find gaps",), text_value="   ")
        run_app(st)
        st.warning.assert_called()
        self.assertIsNone(st.session_state.get("wf"))

    def test_topic_submit_runs_real_pipeline(self):
        st = make_stub(form_clicks=("Discover, analyse and find gaps",), text_value="weather forecasting")
        run_app(st, orchestrator=scripted_orchestrator(FlowLLM()))
        wf = st.session_state["wf"]
        self.assertEqual(wf.topic, "weather forecasting")
        self.assertEqual(wf.completed_stages, ["discover", "index", "analyze_papers", "identify_gaps"])
        self.assertIn("Generate questions", button_labels(st))      # questionnaire offered after analysis
        st.progress.assert_called()

    def test_partial_state_shows_errors_without_crashing(self):
        st = make_stub()
        wf = WorkflowState(topic="x")
        wf.add_error("workflow", "No papers were found for this topic.")
        st.session_state["wf"] = wf
        run_app(st)
        self.assertIn("No papers were found", st.error.call_args.args[0])

    def test_full_state_renders_every_tab(self):
        o, state = flow_state(FlowLLM())
        o.prepare_questions(state)
        o.evaluate_user_work(state, ANSWERS)
        st = make_stub()
        st.session_state["wf"] = state
        run_app(st)
        st.error.assert_not_called()
        self.assertEqual(st.text_area.call_count, len(state.questions))
        st.download_button.assert_called_once()
        st.dataframe.assert_called()
        self.assertGreaterEqual(st.markdown.call_count, 20)

    def test_answer_submission_runs_evaluation(self):
        llm = FlowLLM()
        o, state = flow_state(llm)
        o.prepare_questions(state)
        st = make_stub(form_clicks=("Evaluate my research",))
        st.session_state["wf"] = state
        st.session_state.update(ANSWERS)                           # what the text areas would hold
        run_app(st, orchestrator=scripted_orchestrator(llm))
        self.assertTrue(state.evaluation)
        self.assertEqual(state.errors, [])
        st.success.assert_called()

    def test_submission_with_missing_required_answers_shows_error(self):
        llm = FlowLLM()
        o, state = flow_state(llm)
        o.prepare_questions(state)
        st = make_stub(form_clicks=("Evaluate my research",))
        st.session_state["wf"] = state
        st.session_state["q_title"] = "Only a title"
        run_app(st, orchestrator=scripted_orchestrator(llm))
        self.assertIn("required", st.error.call_args.args[0])
        self.assertEqual(state.evaluation, {})


class TestReportTab(unittest.TestCase):
    @staticmethod
    def downloads(st):
        """file_name -> data for every download button, whether created on st or inside a column."""
        calls = list(st.download_button.call_args_list)
        for col in st.made_columns:
            calls += col.download_button.call_args_list
        return {c.kwargs["file_name"]: c.kwargs["data"] for c in calls}

    def test_generate_button_builds_report_previews_it_and_offers_all_downloads(self):
        llm = ReportLLM()
        o, state = flow_state(llm)
        o.prepare_questions(state)
        o.evaluate_user_work(state, ANSWERS)
        st = make_stub(button_clicks=("Generate report",))
        st.session_state["wf"] = state
        run_app(st, orchestrator=scripted_orchestrator(llm))
        self.assertTrue(state.report)
        self.assertEqual(state.errors, [])
        files = self.downloads(st)
        self.assertEqual(set(files), {"researchpilot_report.pdf", "researchpilot_report.docx",
                                      "researchpilot_report.md", "researchpilot_run.json"})
        self.assertTrue(files["researchpilot_report.pdf"].startswith(b"%PDF"))
        self.assertTrue(files["researchpilot_report.docx"].startswith(b"PK"))
        previews = [c.args[0] for c in st.markdown.call_args_list if c.args and isinstance(c.args[0], str)]
        self.assertTrue(any("## 14. References" in p for p in previews))
        st.spinner.assert_called()

    def test_no_report_means_no_report_downloads(self):
        llm = ReportLLM()
        o, state = flow_state(llm)
        st = make_stub()
        st.session_state["wf"] = state
        run_app(st)
        self.assertEqual(sorted(self.downloads(st)), ["researchpilot_run.json"])

    def test_existing_report_is_shown_without_regenerating(self):
        llm = ReportLLM()
        o, state = flow_state(llm)
        o.generate_report(state, use_llm=False)
        calls_before = len(llm.roles)
        st = make_stub()
        st.session_state["wf"] = state
        run_app(st, orchestrator=scripted_orchestrator(llm))
        self.assertEqual(len(llm.roles), calls_before)                   # no new LLM calls on a plain rerun
        self.assertIn("researchpilot_report.pdf", list(self.downloads(st)))
        shown = [c.args[0] for c in st.warning.call_args_list]
        self.assertTrue(any("not described" in w for w in shown))      # partial-report notice surfaced

    def test_partial_state_tells_user_to_answer_questions_first(self):
        llm = ReportLLM()
        o, state = flow_state(llm)
        st = make_stub()
        st.session_state["wf"] = state
        run_app(st)
        self.assertTrue(any("answer the questions" in str(c.args[0]) for c in st.info.call_args_list))

    def test_report_error_is_shown_not_raised(self):
        llm = ReportLLM()
        o, state = flow_state(llm, max_steps=4)
        st = make_stub(button_clicks=("Generate report",))
        st.session_state["wf"] = state
        run_app(st, orchestrator=lambda **kw: Orchestrator(llm=llm, discover=fake_discover, max_steps=4,
                                                            store_factory=lambda: retrieval.LexicalStore(), **kw))
        self.assertTrue(any("step budget" in c.args[0] for c in st.error.call_args_list))
        self.assertEqual(state.report, {})


if __name__ == "__main__":
    unittest.main()
