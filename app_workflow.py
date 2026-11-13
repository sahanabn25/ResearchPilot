"""ResearchPilot workflow demo (Streamlit). Covers: discovery -> analysis -> gaps -> questionnaire -> evaluation.

Run:  streamlit run app_workflow.py
(The original Week 2 app is still available with: streamlit run app.py)
Option 2 (upload a research paper as a PDF) has its own tab: the uploaded paper is read, compared with related published work, and fed into the same report.
"""
import json
import os

import streamlit as st

from researchpilot import config, llm
from researchpilot.memory import LongTermMemory, ShortTermMemory
from researchpilot.orchestrator import MissingAnswersError, Orchestrator
from researchpilot.reporting import to_docx_bytes, to_markdown, to_pdf_bytes
from researchpilot.reporting.render_md import escape_markdown as esc

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_answers_weather.json")
BADGES = {"verified_from_papers": "🟢 Verified from papers", "user_provided": "🔵 User-provided",
          "ai_generated": "🟠 AI-generated"}
BADGES_UPLOAD = {"verified_from_papers": "🟢 Verified from related papers", "user_provided": "🔵 From uploaded paper",
                 "ai_generated": "🟠 AI-generated"}
STAGES_UPLOAD = [("extract_pdf", "Read the PDF"), ("parse_paper", "Extract method, data, results"),
                 ("discover", "Find related work"), ("index", "Index for retrieval"),
                 ("analyze_papers", "Analyse related papers"), ("identify_gaps", "Identify research gaps"),
                 ("evaluate", "Compare with related work"), ("generate_report", "Generate report")]
STAGES = [("discover", "Discover papers"), ("index", "Index for retrieval"), ("analyze_papers", "Analyse papers"),
          ("identify_gaps", "Identify research gaps"), ("generate_questions", "Ask about your project"),
          ("evaluate", "Compare and evaluate"), ("generate_report", "Generate report")]

st.set_page_config(page_title="ResearchPilot", page_icon="🧭", layout="wide")

# ------------------------------------------------------------------
# RESEARCHPILOT - PROFESSIONAL UI THEME
# Add immediately after st.set_page_config(...)
# ------------------------------------------------------------------

st.markdown("""
<style>

/* ========== GLOBAL THEME ========== */

.stApp {
    background: linear-gradient(135deg, #f7f8ff 0%, #f1f4ff 55%, #f8f7ff 100%);
    color: #202744;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Main headings */

h1, h2, h3 {
    color: #202744 !important;
    font-weight: 750 !important;
    letter-spacing: -0.5px;
}

h1 {
    font-size: 2.5rem !important;
}

h2 {
    font-size: 1.65rem !important;
}

h3 {
    font-size: 1.2rem !important;
}

p, label, .stMarkdown {
    color: #45516d;
}

/* ========== SIDEBAR ========== */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151b3d 0%, #222b58 55%, #343b78 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #c4cbea !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15);
}

[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(90deg, #7566f5, #a16bf5);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 650;
}

/* ========== HERO HEADER ========== */

.rp-hero {
    background: linear-gradient(120deg, #202653 0%, #4644a5 55%, #7761e8 100%);
    padding: 32px 36px;
    border-radius: 20px;
    margin-bottom: 26px;
    box-shadow: 0 12px 35px rgba(65, 60, 155, 0.17);
    position: relative;
    overflow: hidden;
}

.rp-hero h1 {
    color: #ffffff !important;
    font-size: 2.4rem !important;
    margin: 0 0 8px 0;
}

.rp-hero p {
    color: #e0e3ff !important;
    font-size: 1rem;
    margin: 0;
}

.rp-hero .rp-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #ffffff;
    padding: 6px 12px;
    border-radius: 30px;
    font-size: 12px;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* ========== TABS ========== */

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid #e1e5f4;
    padding: 7px 4px;
}

.stTabs [data-baseweb="tab"] {
    background: #ffffff;
    border: 1px solid #e4e7f3;
    border-radius: 10px 10px 0 0;
    padding: 12px 19px;
    color: #5b6480;
    font-weight: 650;
}

.stTabs [aria-selected="true"] {
    background: #eeeeff !important;
    color: #5145c8 !important;
    border-bottom: 3px solid #7162e9 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #7465ed;
}

/* ========== BUTTONS ========== */

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(100deg, #5149d8, #8062ed);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.3rem;
    font-weight: 650;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(94, 78, 211, 0.16);
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: linear-gradient(100deg, #4038bb, #7050d8);
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(94, 78, 211, 0.25);
}

.stButton button:disabled {
    background: #d8dbea;
    color: #7b8297 !important;
    box-shadow: none;
}

/* ========== INPUT FIELDS ========== */

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff;
    border: 1px solid #dce1f0;
    border-radius: 10px;
    color: #252c48;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #7465ed;
    box-shadow: 0 0 0 2px rgba(116,101,237,0.12);
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #aaa4eb;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #7162e9;
    background: #f8f7ff;
}

/* ========== METRIC CARDS ========== */

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e8f4;
    padding: 20px 18px;
    border-radius: 15px;
    box-shadow: 0 5px 18px rgba(35, 43, 90, 0.055);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 9px 24px rgba(65, 60, 155, 0.11);
}

[data-testid="stMetricLabel"] {
    color: #69738f !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #5045c5 !important;
    font-size: 1.9rem !important;
    font-weight: 750;
}

/* ========== EXPANDERS / CARDS ========== */

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e3e7f3;
    border-radius: 12px;
    margin-bottom: 10px;
    box-shadow: 0 3px 12px rgba(35, 43, 90, 0.035);
}

[data-testid="stExpander"] summary {
    font-weight: 650;
    color: #343b67;
}

[data-testid="stExpander"] summary:hover {
    color: #6756d8;
}

/* ========== PROGRESS BAR ========== */

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #5048d6, #a16af0);
    border-radius: 20px;
}

.stProgress > div > div {
    background: #e4e5f5;
    border-radius: 20px;
    height: 11px;
}

/* ========== ALERTS ========== */

[data-testid="stAlert"] {
    border-radius: 11px;
    border: none;
}

.stSuccess {
    background: #e8f8ef;
}

.stWarning {
    background: #fff6df;
}

.stError {
    background: #fff0f0;
}

.stInfo {
    background: #edf2ff;
}

/* ========== DATAFRAMES ========== */

[data-testid="stDataFrame"] {
    border: 1px solid #e2e6f2;
    border-radius: 12px;
    overflow: hidden;
    background: white;
}

/* ========== CHECKBOXES ========== */

.stCheckbox label {
    color: #394363 !important;
    font-weight: 500;
}

/* ========== DOWNLOAD BUTTONS ========== */

.stDownloadButton button {
    width: 100%;
    background: linear-gradient(100deg, #5149d8, #8062ed);
    color: #ffffff !important;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 650;
}

.stDownloadButton button:hover {
    background: linear-gradient(100deg, #4038bb, #7050d8);
}

/* ========== DIVIDERS / CAPTIONS ========== */

hr {
    border-color: #e2e6f2;
}

[data-testid="stCaptionContainer"] {
    color: #78819a;
    font-size: 0.82rem;
}

/* ========== RESPONSIVE DESIGN ========== */

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }

    .rp-hero {
        padding: 22px;
    }

    .rp-hero h1 {
        font-size: 1.8rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 9px 10px;
        font-size: 0.8rem;
    }
}


/* ========== BUTTON TEXT VISIBILITY FIX ========== */

/* Keep text readable on all primary action and download buttons. */
.stButton button,
.stFormSubmitButton button,
.stDownloadButton button,
[data-testid="stSidebar"] .stButton button {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Streamlit may place button labels inside nested paragraph/span elements. */
.stButton button p,
.stButton button span,
.stButton button div,
.stFormSubmitButton button p,
.stFormSubmitButton button span,
.stFormSubmitButton button div,
.stDownloadButton button p,
.stDownloadButton button span,
.stDownloadButton button div,
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button span,
[data-testid="stSidebar"] .stButton button div {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Disabled buttons use dark text on a light background for contrast. */
.stButton button:disabled,
.stFormSubmitButton button:disabled,
.stDownloadButton button:disabled {
    color: #35405C !important;
    -webkit-text-fill-color: #35405C !important;
    background: #D8DBEA !important;
    opacity: 1 !important;
}

.stButton button:disabled p,
.stButton button:disabled span,
.stFormSubmitButton button:disabled p,
.stFormSubmitButton button:disabled span,
.stDownloadButton button:disabled p,
.stDownloadButton button:disabled span {
    color: #35405C !important;
    -webkit-text-fill-color: #35405C !important;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_long_term():
    return LongTermMemory()


def get_short_term():
    if "short_term" not in st.session_state:
        st.session_state["short_term"] = ShortTermMemory()
    return st.session_state["short_term"]


def make_orchestrator(on_progress=None):
    return Orchestrator(long_term=get_long_term(), short_term=get_short_term(), on_progress=on_progress)


def clear_answer_widgets():
    for key in [k for k in st.session_state if str(k).startswith("q_")]:
        del st.session_state[key]


def fill_sample():
    with open(SAMPLE_PATH, encoding="utf-8") as fh:
        for key, value in json.load(fh).items():
            if not key.startswith("_"):
                st.session_state[key] = value


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧭 ResearchPilot")
    st.caption(f"LLM: {llm.describe()}")
    if st.button("Start over"):
        st.session_state.pop("wf", None)
        st.session_state.pop("short_term", None)
        clear_answer_widgets()
    _state = st.session_state.get("wf")
    _done = set(_state.completed_stages) if _state else set()
    st.markdown("**Workflow**")
    for key, label in (STAGES_UPLOAD if _state and _state.mode == "upload" else STAGES):
        st.markdown(f"{'✅' if key in _done else '⬜'} {label}")
    if _state:
        st.caption(f"Steps used: {_state.steps_used}/{config.MAX_STEPS}")


# ------------------------------------------------------------------ run + render helpers
def run_topic(topic: str, limit: int):
    bar = st.progress(0.0, text="Starting…")

    def on_progress(stage, message, fraction):
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

    clear_answer_widgets()
    try:
        state = make_orchestrator(on_progress).run_topic_analysis(topic.strip(), limit=limit)
    except ValueError as exc:
        bar.empty()
        st.error(str(exc))
        return
    bar.empty()
    st.session_state["wf"] = state


def run_upload(uploaded):
    bar = st.progress(0.0, text="Starting…")

    def on_progress(stage, message, fraction):
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

    clear_answer_widgets()
    try:
        state = make_orchestrator(on_progress).run_paper_analysis(uploaded.getvalue(), uploaded.name)
    except Exception as exc:  # the orchestrator records problems itself; this is a last-resort guard
        bar.empty()
        st.error(f"Could not analyse this file: {type(exc).__name__}: {exc}")
        return
    bar.empty()
    st.session_state["wf"] = state


def show_new_errors(state, before: int):
    for err in state.errors[before:]:
        st.error(err["message"])


def title_lookup(state) -> dict:
    lookup = {p.get("paper_id"): p.get("title", "") for p in state.papers}
    lookup.update({pid: a.get("title", "") for pid, a in state.analyses.items()})
    return lookup


def render_uploaded_paper(state):
    p = state.paper
    if not p:
        st.info("Upload a paper in the Option 2 tab to begin.")
        return
    parsed = p.get("parsed", {})
    cols = st.columns(4)
    cols[0].metric("Pages read", f"{p.get('pages_read')}/{p.get('n_pages')}")
    cols[1].metric("Sections found", len(p.get("section_order", [])))
    cols[2].metric("Tables", len(p.get("tables", [])))
    cols[3].metric("Equation-like lines", len(p.get("equations", [])))
    st.markdown(f"**{esc(state.topic)}**  \n{esc(p.get('filename', ''))}")
    for note in parsed.get("notes", []):
        st.warning(esc(note))
    for warning in p.get("warnings", []):
        st.caption(f"Extraction note: {esc(warning)}")
    st.subheader("What the system extracted")
    labels = [("problem", "Research problem"), ("objectives", "Objectives"), ("methodology", "Methodology"),
              ("techniques", "Techniques"), ("datasets", "Datasets"), ("metrics", "Metrics reported"),
              ("results", "Results"), ("conclusions", "Conclusions"), ("limitations", "Limitations stated"),
              ("future_work", "Future work stated"), ("advantages", "Claimed contributions")]
    fields, sources = state.profile.get("fields", {}), parsed.get("fields_source", {})
    for key, label in labels:
        if fields.get(key):
            st.markdown(f"**{label}** ({sources.get(key, 'extracted')}): {esc(fields[key])}")
    dropped = (parsed.get("claims") or {}).get("dropped") or {}
    if dropped:
        st.caption(f"{sum(len(v) for v in dropped.values())} AI-extracted item(s) were removed because the paper's text "
                   "did not support them.")
    with st.expander("Abstract"):
        st.write(esc(p.get("abstract", "")))
    tables = [t for t in p.get("tables", []) if t.get("rows")]
    if tables:
        with st.expander(f"Tables detected ({len(tables)})"):
            for t in tables:
                st.markdown(f"**{esc(t.get('caption') or 'Table on page ' + str(t['page']))}** (page {t['page']}, {t['source']})")
                st.dataframe(t["rows"])
    if p.get("equations"):
        with st.expander(f"Equation-like lines ({len(p['equations'])})"):
            st.code("\n".join(p["equations"]))
    if p.get("figures"):
        with st.expander(f"Figures (captions only, {len(p['figures'])})"):
            st.markdown("\n".join(f"- {esc(f['label'])}: {esc(f['caption'])}" for f in p["figures"]))
    if parsed.get("queries"):
        st.caption("Searches used to find related work: " + " | ".join(esc(q) for q in parsed["queries"]))


def render_papers(state):
    if not state.papers:
        st.info("No papers yet.")
        return
    for i, p in enumerate(state.papers, 1):
        with st.expander(f"{i}. {esc(p.get('title', 'Untitled'))} ({p.get('year') or 'n.d.'})"):
            st.markdown(f"**Authors:** {esc(', '.join(p.get('authors') or []) or 'unavailable')}")
            st.markdown(f"**Venue:** {esc(p.get('venue') or 'unavailable')}  |  **Found via:** {esc(', '.join(p.get('sources', [])))}")
            st.markdown(f"**DOI:** {esc(p.get('doi') or 'unavailable')}")
            link = p.get("url") or p.get("oa_url")
            if isinstance(link, str) and link.startswith("http"):
                st.markdown(f"[Open source]({link.replace('(', '%28').replace(')', '%29').replace(' ', '%20')})")
            st.markdown("**Abstract:**")
            st.write(esc(p.get("abstract") or "No abstract available, so this paper is not analysed."))


def render_analysis(state):
    if not state.analyses:
        st.info("Run the analysis first.")
        return
    st.caption(f"Analysis stats: {state.analysis_stats}")
    skipped = []
    for a in state.analyses.values():
        if a["status"] != "ok":
            skipped.append(f"- {esc(a.get('title', '')[:90])}: {a['status']}" + (f" — {esc(a['error'])}" if a.get("error") else ""))
            continue
        with st.expander(f"{esc(a['title'])} ({a.get('year') or 'n.d.'})"):
            st.markdown(f"**Summary (AI paraphrase):** {esc(a['summary'])}")
            for label, key in (("Research problem", "research_problem"), ("Methodology", "methodology"),
                               ("Key results (verified against abstract)", "key_results"),
                               ("Stated limitations", "stated_limitations")):
                st.markdown(f"**{label}:** {esc(a[key])}")
            for label, key in (("Techniques", "techniques"), ("Datasets", "datasets"), ("Metrics", "metrics")):
                st.markdown(f"**{label}:** {esc(', '.join(a[key]) or 'not reported')}")
            v = a.get("verification", {})
            if v.get("dropped_items") or v.get("unverified_numbers"):
                st.caption(f"Removed because the abstract did not support them: {esc(str(v))}")
    if skipped:
        with st.expander("Papers not analysed"):
            st.markdown("\n".join(skipped))


def render_gaps(state):
    g = state.gaps
    if not g:
        st.info("Run the analysis first.")
        return
    if g.get("insufficient_evidence"):
        st.warning(g["message"])
        return
    st.caption(g["scope_note"])
    titles = title_lookup(state)
    for heading, key, text_key in (("Trends", "trends", "trend"),
                                   ("Common limitations", "common_limitations", "limitation")):
        st.subheader(heading)
        for item in g[key]:
            st.markdown(f"- {esc(item[text_key])}  \n  *Supported by {len(item['paper_ids'])} paper(s) "
                        f"({item['evidence_strength']})*")
    st.subheader("Research gaps")
    for item in g["gaps"]:
        st.markdown(f"**{esc(item['gap'])}**")
        st.caption(f"Evidence strength: {item['evidence_strength']} ({len(item['paper_ids'])} paper(s)). "
                   f"{esc(item.get('rationale', ''))}")
        with st.expander("Supporting papers and closest passages"):
            for pid in item["paper_ids"]:
                st.markdown(f"- {esc(titles.get(pid, pid))}")
            for ev in item.get("related_evidence", []):
                st.caption(f"{esc(ev['title'][:80])} (score {ev['score']}): {esc(ev['snippet'])}")
    if g.get("dropped_unsupported"):
        st.caption(f"{g['dropped_unsupported']} model suggestion(s) removed for citing no valid paper.")


def render_questions(state):
    if not state.analyses or not state.gaps:
        st.info("Complete the literature analysis first.")
        return
    if not state.questions:
        st.write("Next, ResearchPilot asks about **your own project**. Questions are tailored to the gaps found.")
        if st.button("Generate questions", type="primary"):
            before = len(state.errors)
            with st.spinner("Preparing questions…"):
                make_orchestrator().prepare_questions(state)
            show_new_errors(state, before)
    if not state.questions:
        return
    st.button("Fill with FICTIONAL sample answers (demo only)", on_click=fill_sample)
    with st.form("answers_form"):
        for q in state.questions:
            if q["kind"] == "gap":
                st.caption(f"Literature gap: {esc(q['gap'])}")
            st.text_area(q["question"] + (" (required)" if q["required"] else ""), key=q["id"], height=90)
        submitted = st.form_submit_button("Evaluate my research", type="primary")
    if submitted:
        answers = {q["id"]: st.session_state.get(q["id"], "") for q in state.questions}
        before = len(state.errors)
        try:
            with st.spinner("Comparing your work with the literature…"):
                make_orchestrator().evaluate_user_work(state, answers)
        except MissingAnswersError as exc:
            st.error(str(exc))
            return
        show_new_errors(state, before)
        if state.evaluation:
            st.success("Evaluation complete. Open the Evaluation tab.")


def render_evaluation(state):
    ev = state.evaluation
    if not ev:
        st.info("Answer the questions to see the evaluation.")
        return
    titles = title_lookup(state)
    upload = state.mode == "upload"
    BADGES = BADGES_UPLOAD if upload else globals()["BADGES"]
    if ev.get("message"):
        st.warning(ev["message"])
    legend = dict(ev["provenance_legend"])
    if upload:
        legend["user_provided"] = "Stated in the uploaded paper; not independently verified."
    st.caption("Legend: " + "  |  ".join(f"{BADGES[k]}: {v}" for k, v in legend.items()))

    st.subheader("Checked by code")
    e = ev["evidence"]
    for c in e["improvement_checks"]:
        chk = c["check"]
        who = "Paper's claim" if upload else "Your claim"
        st.markdown(f"**{who}:** {esc(c['claim'])}  \n{BADGES['user_provided']} → **{chk['status'].replace('_', ' ')}**"
                    + (f" ({chk['reason']})" if chk.get("reason") else "")
                    + (f", relative change {chk['relative_change_pct']}%" if chk.get("relative_change_pct") is not None else ""))
        if chk.get("caveat"):
            st.caption(esc(chk["caveat"]))
    rows = [{"metric": r["metric"], "your value": r["user_value"], "paper value": r["paper_value"],
             "paper": titles.get(r["paper_id"], r["paper_id"])[:70], "yours better": r["user_value_better"]}
            for r in e["numeric_comparisons"]["comparisons"]]
    st.markdown(f"**Same-dataset numeric comparisons** ({BADGES['verified_from_papers']})")
    if rows:
        st.dataframe(rows)
    else:
        st.caption("No paper stated a comparable value for the same dataset and metric.")
    st.caption(e["numeric_comparisons"]["note"])
    st.markdown("**Technique / dataset found in retrieved papers** (word-level match)")
    for group in ("technique_prior_use", "dataset_overlap"):
        for row in e[group]:
            names = "; ".join(titles.get(p, p)[:50] for p in row["paper_ids"]) or "none found"
            st.markdown(f"- {esc(row['item'])}: {esc(names)}")
    st.markdown("**Rigor elements mentioned in your answers**")
    st.markdown("  \n".join(f"{'✔' if v else '✖'} {k.replace('_', ' ')}" for k, v in e["rigor"].items()))
    st.caption(e["rigor_note"])

    a = ev.get("assessment") or {}
    if a:
        st.subheader("Assessment")
        for g in a["gap_alignment"]:
            st.markdown(f"**Gap:** {esc(g['gap'])}  \n{BADGES[g['provenance']]} → **{g['addressed']}**. {esc(g['rationale'])}")
            if g["paper_ids"]:
                st.caption("Cited: " + esc("; ".join(titles.get(p, p)[:60] for p in g["paper_ids"])))
        md = a["methodology_difference"]
        st.markdown(f"**Methodology vs existing work:** {BADGES[md['provenance']]} → **{md['assessment']}**. {esc(md['rationale'])}")
        for heading, key, text_key in (("Remaining limitations", "remaining_limitations", "limitation"),
                                       ("Suggested experiments", "suggested_experiments", "experiment"),
                                       ("Potential contributions", "potential_contributions", "contribution")):
            st.markdown(f"**{heading}**")
            for item in a[key]:
                st.markdown(f"- {esc(item[text_key])}  \n  {BADGES[item['provenance']]}")
                if item.get("caveat"):
                    st.warning(esc(item["caveat"]))
    st.caption(f"Items removed as unsupported: {ev['dropped_unsupported']}. {ev['scope_note']}")


def render_report(state):
    if not (state.analyses or state.papers):
        st.info("Run the analysis first.")
        return
    if not state.evaluation:
        st.info("The evaluation did not complete, so the report will be partial. You can still generate it."
                if state.mode == "upload" else
                "For a complete report, answer the questions in the 'Your project' tab first. "
                "You can also generate a partial report now.")
    use_llm = st.checkbox("Let the AI write the literature-review and conclusion paragraphs "
                          "(one extra LLM call; output is validated)", value=config.REPORT_USE_LLM, key="report_use_llm")
    if st.button("Generate report", type="primary"):
        before = len(state.errors)
        with st.spinner("Writing the report…"):
            make_orchestrator().generate_report(state, use_llm=use_llm)
        show_new_errors(state, before)
    report = state.report
    if not report:
        return
    st.caption(f"Generated {report['generated_at']}. Regenerate after changing your answers.")
    for notice in report.get("notices", []):
        st.warning(notice)
    try:
        pdf, docx, md = to_pdf_bytes(report), to_docx_bytes(report), to_markdown(report)
    except Exception as exc:
        st.error(f"Could not prepare the downloads: {type(exc).__name__}: {exc}")
        return
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download PDF", data=pdf, file_name="researchpilot_report.pdf", mime="application/pdf")
    c2.download_button("Download Word (DOCX)", data=docx, file_name="researchpilot_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    c3.download_button("Download Markdown", data=md, file_name="researchpilot_report.md", mime="text/markdown")
    st.divider()
    st.markdown(to_markdown(report, safe=True))        # preview: untrusted text must not render as links/images


def render_trace(state):
    st.caption(f"Steps used: {state.steps_used}/{config.MAX_STEPS}  |  Stages: {', '.join(state.completed_stages)}")
    st.caption(f"Sources: {state.source_status}")
    st.dataframe(state.events)
    st.download_button("Download full run (JSON)", data=json.dumps(state.to_dict(), indent=2, ensure_ascii=False),
                       file_name="researchpilot_run.json", mime="application/json")


def render_results(state):
    cols = st.columns(4)
    n_abs = sum(1 for p in state.papers if len((p.get("abstract") or "").strip()) >= 80)
    cols[0].metric("Related papers found" if state.mode == "upload" else "Papers found", len(state.papers))
    cols[1].metric("With abstracts", n_abs)
    cols[2].metric("Analysed", state.analysis_stats.get("analysed", 0) + state.analysis_stats.get("reused", 0))
    cols[3].metric("Research gaps", len(state.gaps.get("gaps", [])) if state.gaps else 0)
    st.caption(f"Sources: {state.source_status}")
    for err in state.errors:
        st.error(err["message"])
    if state.mode == "upload":
        names = ["1 Uploaded paper", "2 Related papers", "3 Analysis", "4 Gaps", "5 Evaluation", "6 Report", "Trace"]
        renderers = (render_uploaded_paper, render_papers, render_analysis, render_gaps, render_evaluation,
                     render_report, render_trace)
    else:
        names = ["1 Papers", "2 Analysis", "3 Gaps", "4 Your project", "5 Evaluation", "6 Report", "Trace"]
        renderers = (render_papers, render_analysis, render_gaps, render_questions, render_evaluation,
                     render_report, render_trace)
    tabs = st.tabs(names)
    for tab, fn in zip(tabs, renderers):
        with tab:
            fn(state)


# ------------------------------------------------------------------ page

st.markdown("""
<div class="rp-hero">
    <div class="rp-tag">✦ AI-POWERED RESEARCH INTELLIGENCE</div>
    <h1>🧭 ResearchPilot</h1>
    <p>
        Your intelligent research companion.
        Discover literature, uncover research gaps,
        evaluate methodologies and generate research reports.
    </p>
</div>
""", unsafe_allow_html=True)

if not llm.is_configured():
    st.error(llm.missing_credentials_message() + " Then restart the app.")
    st.stop()

option1, option2 = st.tabs(["Option 1: Start New Research Analysis", "Option 2: Upload Research Paper"])
with option1:
    with st.form("topic_form"):
        topic = st.text_input("Research topic or project idea", placeholder="e.g. Machine learning for weather forecasting")
        limit = st.selectbox("Papers to retrieve", [5, 10, 15], index=1)
        submitted_topic = st.form_submit_button("Discover, analyse and find gaps", type="primary")
    if submitted_topic:
        if topic.strip():
            run_topic(topic, limit)
        else:
            st.warning("Enter a research topic to continue.")
with option2:
    st.markdown("Upload an existing research paper. ResearchPilot reads it, finds related published work, and runs the "
                "same gap analysis, comparison and report as Option 1. The paper plays the role of your own project.")
    uploaded = st.file_uploader(f"Research paper (text-based PDF, up to {config.MAX_PDF_MB} MB)", type=["pdf"], key="paper_upload")
    st.caption("Scanned PDFs are not supported (no OCR). Only text and detected tables and equations are read; "
               "data inside graphs is not.")
    if st.button("Analyse this paper", type="primary", disabled=uploaded is None):
        run_upload(uploaded)

_wf = st.session_state.get("wf")
if _wf:
    st.divider()
    render_results(_wf)
else:
    st.caption("Enter a topic above to begin. Searching and analysing takes about a minute.")
