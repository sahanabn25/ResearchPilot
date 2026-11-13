"""Orchestrator for the topic workflow (discover -> analyse -> identify gaps).

What makes this agentic rather than a fixed script
- It inspects intermediate results and DECIDES: if too few retrieved papers carry
  abstracts, it asks the LLM for alternative queries and searches again (once).
- It has a hard step budget (MAX_STEPS) and explicit stop rules (no papers, no
  abstracts, budget exhausted) instead of looping or crashing.
- Stage failures are isolated: earlier results stay in the shared state so the UI
  can still show papers if analysis fails, and the failure is recorded in the trace.
"""
from typing import Callable, Optional

from . import config, retrieval
from .agents.comparison import evaluate_user_work, structure_profile  # noqa: F401
from .agents.gap_agent import identify_gaps
from .agents.interaction import build_profile, generate_questions, validate_answers
from .agents.report_writer import write_report
from .agents.paper_parser import is_same_paper, parse_paper, search_queries
from .agents.paper_analysis import analyze_papers, has_usable_abstract
from .discovery import DiscoveryError, discover_papers, merge_papers
from .llm import generate_json
from .logging_setup import get_logger
from .memory import LongTermMemory, ShortTermMemory
from .state import WorkflowState
from .tools.pdf_reader import PdfError, extract_pdf

log = get_logger(__name__)

REFORMULATE_INSTRUCTIONS = """You propose alternative search queries for scholarly databases.
Given a research topic, return 2 different queries that would surface relevant published papers
(use standard terminology, synonyms or a broader/narrower framing). Do not repeat the original.
Return valid JSON only: {"queries": ["...", "..."]}"""


class StepLimitExceeded(Exception):
    pass


class MissingAnswersError(ValueError):
    """Required questionnaire answers are missing; `.missing` lists the question ids."""

    def __init__(self, missing: list):
        super().__init__("Please answer the required questions: " + ", ".join(missing))
        self.missing = missing


class _Stop(Exception):
    """Expected early stop (e.g. nothing to analyse). Recorded as an error in the state."""


class Orchestrator:
    def __init__(self, llm: Callable = generate_json, discover: Callable = discover_papers,
                 store_factory: Callable = retrieval.get_store,
                 long_term: Optional[LongTermMemory] = None, short_term: Optional[ShortTermMemory] = None,
                 on_progress: Optional[Callable] = None, max_steps: Optional[int] = None):
        self.llm = llm
        self.discover = discover
        self.store_factory = store_factory
        self.long_term = long_term
        self.short_term = short_term
        self.on_progress = on_progress
        self.max_steps = max_steps or config.MAX_STEPS

    # ---- helpers ----
    def _progress(self, stage: str, message: str, fraction: float) -> None:
        if self.on_progress:
            try:
                self.on_progress(stage, message, fraction)
            except Exception:  # a UI callback bug must never kill the workflow
                log.warning("progress callback raised; ignored")

    def _step(self, state: WorkflowState, name: str) -> None:
        if state.steps_used >= self.max_steps:
            raise StepLimitExceeded(f"step budget of {self.max_steps} reached before '{name}'")
        state.steps_used += 1
        state.log(name, f"step {state.steps_used}/{self.max_steps}")

    @staticmethod
    def _abstract_count(papers: list) -> int:
        return sum(1 for p in papers if has_usable_abstract(p))

    # ---- stages ----
    def _stage_discover(self, state: WorkflowState, limit: int) -> None:
        self._step(state, "discover")
        self._progress("discover", "Searching Crossref, OpenAlex and arXiv…", 0.05)
        try:
            result = self.discover(state.topic, limit, memory=self.long_term)
        except DiscoveryError as exc:
            raise _Stop(f"All paper sources failed: {exc}") from None
        state.papers = result["papers"]
        state.source_status = result["source_status"]
        state.log("discover", f"{len(state.papers)} papers, {self._abstract_count(state.papers)} with abstracts "
                              f"(cache={result.get('from_cache')}); sources: {state.source_status}")
        if not state.papers:
            raise _Stop("No papers were found for this topic. Try different or broader wording.")
        state.completed_stages.append("discover")

    def _reformulate(self, topic: str) -> list:
        data = self.llm(f"Research topic: {topic}", system_instruction=REFORMULATE_INSTRUCTIONS, temperature=0.4)
        queries = data.get("queries") if isinstance(data, dict) else None
        if not isinstance(queries, list):
            raise ValueError("model returned no 'queries' list")
        clean = []
        for q in queries:
            if isinstance(q, str) and 3 <= len(q.strip()) <= 200 and q.strip().lower() != topic.lower() \
                    and q.strip() not in clean:
                clean.append(q.strip())
        return clean[:2]

    def _stage_broaden(self, state: WorkflowState, limit: int) -> None:
        if self._abstract_count(state.papers) >= config.MIN_ABSTRACT_PAPERS:
            return
        self._step(state, "broaden_search")
        self._progress("discover", "Few abstracts found; trying alternative queries…", 0.15)
        try:
            queries = self._reformulate(state.topic)
        except Exception as exc:  # LLM unavailable: continue with what we have
            state.log("broaden_search", f"skipped, could not reformulate ({type(exc).__name__})")
            return
        lists = [state.papers]
        for q in queries:
            try:
                lists.append(self.discover(q, limit, memory=self.long_term)["papers"])
            except (DiscoveryError, ValueError) as exc:
                state.log("broaden_search", f"query {q!r} failed ({type(exc).__name__})")
        merged = merge_papers(lists)
        merged.sort(key=lambda p: not has_usable_abstract(p))
        state.papers = merged[:min(limit * 2, config.MAX_PAPERS)]
        state.log("broaden_search", f"queries {queries}; now {len(state.papers)} papers, "
                                    f"{self._abstract_count(state.papers)} with abstracts")

    def _stage_index(self, state: WorkflowState) -> None:
        self._step(state, "index")
        self._progress("index", "Indexing papers for retrieval…", 0.3)
        try:
            state.store = self.store_factory()
            n = retrieval.index_papers(state.store, state.papers)
            state.log("index", f"{n} chunks indexed ({getattr(state.store, 'backend_name', 'store')})")
            state.completed_stages.append("index")
        except Exception as exc:  # gaps can still be found without the index
            state.store = None
            state.log("index", f"skipped, index unavailable ({type(exc).__name__})")

    def _stage_analyze(self, state: WorkflowState) -> None:
        if self._abstract_count(state.papers) == 0:
            raise _Stop("None of the retrieved papers has an abstract, so there is nothing reliable to analyse. "
                        "Try a different topic or upload the paper PDFs.")
        self._step(state, "analyze_papers")
        self._progress("analysis", "Analysing papers…", 0.4)
        result = analyze_papers(state.papers, existing=state.analyses, llm=self.llm,
                                on_progress=lambda s, m, f: self._progress(s, m, 0.4 + 0.3 * f))
        state.analyses, state.analysis_stats = result["analyses"], result["stats"]
        state.log("analyze_papers", f"stats: {state.analysis_stats}")
        if state.analysis_stats["analysed"] + state.analysis_stats["reused"] == 0:
            reason = next((a["error"] for a in state.analyses.values() if a.get("error")), "")
            raise _Stop("Paper analysis did not succeed for any paper"
                        + (f". First failure: {reason}" if reason else " (see the trace for details)."))
        state.completed_stages.append("analyze_papers")

    def _stage_gaps(self, state: WorkflowState) -> None:
        self._step(state, "identify_gaps")
        self._progress("gaps", "Identifying trends, limitations and research gaps…", 0.8)
        state.gaps = identify_gaps(state.topic, state.analyses, store=state.store, llm=self.llm)
        state.log("identify_gaps", f"{len(state.gaps['gaps'])} gaps, {len(state.gaps['trends'])} trends, "
                                   f"dropped_unsupported={state.gaps['dropped_unsupported']}")
        state.completed_stages.append("identify_gaps")

    # ---- public API ----
    def run_topic_analysis(self, topic: str, limit: Optional[int] = None) -> WorkflowState:
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Research topic cannot be empty.")
        limit = limit or config.DISCOVERY_LIMIT
        state = WorkflowState(mode="topic", topic=topic.strip())
        if self.short_term:
            self.short_term.set("topic", state.topic)

        try:
            self._stage_discover(state, limit)
            self._stage_broaden(state, limit)
            self._stage_index(state)
            self._stage_analyze(state)
            self._stage_gaps(state)
            self._progress("done", "Analysis complete.", 1.0)
        except _Stop as exc:
            state.add_error("workflow", str(exc))
        except StepLimitExceeded as exc:
            state.add_error("workflow", f"Stopped: {exc}")
        except Exception as exc:  # unexpected: keep partial results, never crash the UI
            log.exception("workflow stage failed")
            state.add_error("workflow", f"{type(exc).__name__}: {exc}"[:300])

        self._remember(state)
        return state

    # ---- Option 2: uploaded paper -----------------------------------------------------------------
    def _stage_extract(self, state: WorkflowState, pdf_bytes, filename: str) -> None:
        self._step(state, "extract_pdf")
        self._progress("extract", "Reading the PDF…", 0.05)
        try:
            state.paper = extract_pdf(pdf_bytes, filename)
        except PdfError as exc:
            raise _Stop(str(exc)) from None
        state.paper["text"] = state.paper["text"][:config.MAX_STORED_TEXT]
        state.topic = state.paper.get("title_guess") or state.paper["filename"]
        p = state.paper
        state.log("extract_pdf", f"{p['pages_read']}/{p['n_pages']} pages, {p['n_chars']} characters, sections={p['section_order']}, "
                                 f"tables={len(p['tables'])}, equations={len(p['equations'])}, figures={len(p['figures'])}")
        state.completed_stages.append("extract_pdf")

    def _stage_parse(self, state: WorkflowState) -> None:
        self._step(state, "parse_paper")
        self._progress("parse", "Extracting the paper's problem, method and results…", 0.15)
        parsed = parse_paper(state.paper, llm=self.llm)
        state.profile = parsed["profile"]
        state.paper["parsed"] = {k: parsed[k] for k in ("title", "claims", "future_work", "conclusions", "fields_source",
                                                        "notes", "truncated", "llm_calls", "source")}
        state.topic = parsed["title"]
        state.log("parse_paper", f"source={parsed['source']}, llm_calls={parsed['llm_calls']}, "
                                 f"techniques={len(parsed['claims']['techniques'])}, metrics={len(parsed['claims']['metrics'])}")
        state.completed_stages.append("parse_paper")

    def _without_self(self, state: WorkflowState, papers: list) -> list:
        title, doi = state.paper.get("parsed", {}).get("title", state.topic), state.paper.get("doi")
        kept = [p for p in papers if not is_same_paper(p, title, doi)]
        if len(kept) != len(papers):
            state.log("discover", f"excluded {len(papers) - len(kept)} record(s) matching the uploaded paper itself")
        return kept

    def _stage_discover_related(self, state: WorkflowState, limit: int) -> None:
        self._step(state, "discover")
        self._progress("discover", "Searching for related published work…", 0.3)
        queries = search_queries({**state.paper["parsed"], "profile": state.profile}, state.paper)
        state.paper["parsed"]["queries"] = queries
        lists, status = [], {}
        for q in queries:
            try:
                result = self.discover(q, limit, memory=self.long_term)
            except (DiscoveryError, ValueError) as exc:
                state.log("discover", f"query {q!r} failed ({type(exc).__name__})")
                continue
            lists.append(result["papers"])
            status.update(result["source_status"])
        if not lists:
            raise _Stop("All paper sources failed, so related work could not be searched. Try again later.")
        merged = self._without_self(state, merge_papers(lists))
        merged.sort(key=lambda p: not has_usable_abstract(p))
        state.papers, state.source_status = merged[:min(limit * 2, config.MAX_PAPERS)], status
        state.log("discover", f"{len(queries)} queries -> {len(state.papers)} related papers, "
                              f"{self._abstract_count(state.papers)} with abstracts")
        if not state.papers:
            raise _Stop("No related papers were found for this paper. The analysis of the paper itself is shown.")
        state.completed_stages.append("discover")

    def _stage_evaluate_paper(self, state: WorkflowState) -> None:
        self._step(state, "evaluate_paper")
        self._progress("evaluation", "Comparing the paper with related work…", 0.9)
        parsed = state.paper["parsed"]
        state.evaluation = evaluate_user_work(state.topic, state.profile, state.analyses, state.gaps, llm=self.llm,
                                              claims=parsed["claims"], source_text=state.paper["text"])
        state.log("evaluate_paper", f"literature_available={state.evaluation['literature_available']}, "
                                    f"dropped_unsupported={state.evaluation['dropped_unsupported']}")
        state.completed_stages.append("evaluate")

    def run_paper_analysis(self, pdf_bytes, filename: str = "paper.pdf", limit: Optional[int] = None) -> WorkflowState:
        """Option 2: analyse an uploaded paper (PDF). The paper becomes the subject of the same comparison and
        report pipeline as Option 1: extract -> parse -> find related work -> analyse -> gaps -> evaluate.
        Problems are recorded in the state (never raised), with whatever was completed kept."""
        limit = limit or config.DISCOVERY_LIMIT
        state = WorkflowState(mode="upload")
        try:
            self._stage_extract(state, pdf_bytes, filename)
            self._stage_parse(state)
            self._stage_discover_related(state, limit)
            self._stage_broaden(state, limit)
            state.papers = self._without_self(state, state.papers)
            self._stage_index(state)
            self._stage_analyze(state)
            self._stage_gaps(state)
            self._stage_evaluate_paper(state)
            self._progress("done", "Analysis complete.", 1.0)
        except _Stop as exc:
            state.add_error("workflow", str(exc))
        except StepLimitExceeded as exc:
            state.add_error("workflow", f"Stopped: {exc}")
        except Exception as exc:  # unexpected: keep partial results, never crash the UI
            log.exception("paper workflow stage failed")
            state.add_error("workflow", f"{type(exc).__name__}: {exc}"[:300])
        if self.short_term and state.paper:
            self.short_term.set("topic", state.topic)
        self._remember(state)
        return state

    # ---- interactive part of the workflow (runs after the user has seen the literature analysis) ----
    def prepare_questions(self, state: WorkflowState) -> list:
        """Generate the dynamic questionnaire from the topic and identified gaps (1 step, at most 1 LLM call)."""
        if not state.topic:
            raise ValueError("Run the topic analysis before preparing questions.")
        try:
            self._step(state, "generate_questions")
        except StepLimitExceeded as exc:
            state.add_error("questionnaire", f"Stopped: {exc}")
            return state.questions
        self._progress("questions", "Preparing questions about your project…", 0.0)
        result = generate_questions(state.topic, state.gaps, state.analyses, llm=self.llm)
        state.questions = result["questions"]
        state.log("generate_questions", f"{len(state.questions)} questions (source={result['source']}) {result['note']}")
        state.completed_stages.append("generate_questions")
        if self.short_term:
            self.short_term.add_turn("assistant", f"Asked {len(state.questions)} questions about the user's project.")
        return state.questions

    def evaluate_user_work(self, state: WorkflowState, answers: dict) -> WorkflowState:
        """Validate answers, then structure claims and compare them with the literature (2 steps, 2 LLM calls).

        Raises MissingAnswersError (before spending any step or LLM call) if required answers are missing.
        Other failures are recorded in the state; the answers are kept so the user can retry.
        """
        if not state.questions:
            raise ValueError("Prepare the questionnaire before submitting answers.")
        checked = validate_answers(state.questions, answers)
        if checked["missing_required"]:
            raise MissingAnswersError(checked["missing_required"])
        state.answers = checked["answers"]
        state.profile = build_profile(state.questions, state.answers)
        state.report = {}                      # a report built from earlier answers would now be stale
        if self.short_term:
            self.short_term.add_turn("user", f"Answered {state.profile['n_answered']} questions.")
            self.short_term.set("project_title", state.profile["fields"].get("title", ""))
        try:
            self._step(state, "structure_claims")
            self._step(state, "evaluate")
            self._progress("evaluation", "Comparing your work with the literature…", 0.5)
            state.evaluation = evaluate_user_work(state.topic, state.profile, state.analyses, state.gaps,
                                                  llm=self.llm)
            state.log("evaluate", f"literature_available={state.evaluation['literature_available']}, "
                                  f"dropped_unsupported={state.evaluation['dropped_unsupported']}")
            state.completed_stages.append("evaluate")
            self._progress("done", "Evaluation complete.", 1.0)
        except StepLimitExceeded as exc:
            state.add_error("evaluation", f"Stopped: {exc}")
        except Exception as exc:
            log.exception("evaluation failed")
            state.add_error("evaluation", f"{type(exc).__name__}: {exc}"[:300])
        return state

    def generate_report(self, state: WorkflowState, use_llm=None) -> dict:
        """Build the 14-section research analysis report (1 step, at most 1 LLM call for validated narrative).

        Works after the topic analysis alone (sections about the user's project then say what is missing) and
        is most complete after the evaluation. Failures are recorded in the state, never raised, except for
        an empty state (ValueError).
        """
        if not (state.topic or state.papers or state.analyses):
            raise ValueError("There is nothing to report yet. Run the topic analysis first.")
        try:
            self._step(state, "generate_report")
        except StepLimitExceeded as exc:
            state.add_error("report", f"Stopped: {exc}")
            return state.report
        self._progress("report", "Writing the report…", 0.5)
        try:
            state.report = write_report(state, llm=self.llm, use_llm=use_llm)
            state.log("generate_report", f"narrative={state.report.get('narrative_source')}, "
                                         f"complete={state.report.get('complete')}")
            state.completed_stages.append("generate_report")
            self._progress("done", "Report ready.", 1.0)
        except Exception as exc:
            log.exception("report generation failed")
            state.add_error("report", f"{type(exc).__name__}: {exc}"[:300])
        return state.report

    def _remember(self, state: WorkflowState) -> None:
        if self.short_term:
            self.short_term.set("papers", [p.get("title", "") for p in state.papers])
            self.short_term.set("gaps", [g["gap"] for g in state.gaps.get("gaps", [])])
        if self.long_term and state.papers:
            try:
                self.long_term.save_run(state.mode, state.topic, {
                    "papers": len(state.papers), "analysed": state.analysis_stats.get("analysed", 0),
                    "gaps": len(state.gaps.get("gaps", [])), "errors": len(state.errors),
                    "steps": state.steps_used})
            except Exception as exc:
                log.warning("could not save run: %s", type(exc).__name__)
