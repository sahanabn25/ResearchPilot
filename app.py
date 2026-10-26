import json
import time
from datetime import datetime

import streamlit as st

# Reuse the existing backend functions from agent.py
from agent import search_papers, create_research_plan, GEMINI_API_KEY

st.set_page_config(
    page_title="ResearchPilot | AI Research Workspace",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #667085;
        --line: #e6eaf2;
        --panel: #ffffff;
        --lavender: #7667f5;
        --mint: #dff8ee;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 88% 3%, rgba(118,103,245,.12), transparent 24rem),
            radial-gradient(circle at 5% 20%, rgba(34,197,160,.08), transparent 25rem),
            #f6f8fc;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: rgba(246,248,252,.85); }
    [data-testid="stSidebar"] {
        background: #11182b;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #f3f5ff !important; }
    [data-testid="stSidebar"] .stCaption { color: #aab4ce !important; }
    .brand {
        display:flex; align-items:center; gap:12px; margin: 4px 0 28px 0;
    }
    .brand-mark {
        width:44px; height:44px; display:flex; align-items:center; justify-content:center;
        border-radius:14px; font-size:22px;
        background: linear-gradient(135deg,#8c7cff,#5a4be7);
        box-shadow: 0 8px 24px rgba(118,103,245,.35);
    }
    .brand-title { font:800 21px Manrope,sans-serif; letter-spacing:-.6px; }
    .brand-sub { color:#aab4ce; font-size:11px; margin-top:1px; letter-spacing:1.7px; text-transform:uppercase; }
    .side-label { color:#9aa7c6; font-size:10px; font-weight:700; letter-spacing:1.6px; margin:24px 0 8px; }
    .side-card {
        border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.055);
        padding:14px; border-radius:15px; margin-top:18px;
    }
    .side-card-title { font-size:12px; color:#b9c4df; margin-bottom:7px; }
    .side-card-value { font-size:14px; font-weight:700; }
    .hero {
        background: linear-gradient(120deg,#171f39 0%,#282b59 54%,#5747bc 100%);
        color:white; border-radius:24px; padding:32px 34px; position:relative; overflow:hidden;
        box-shadow:0 18px 42px rgba(35,38,82,.16); margin: 4px 0 22px;
    }
    .hero:after {
        content:""; position:absolute; width:290px; height:290px; right:-75px; top:-120px;
        border:1px solid rgba(255,255,255,.18); border-radius:50%;
        box-shadow:0 0 0 28px rgba(255,255,255,.035),0 0 0 58px rgba(255,255,255,.025);
    }
    .eyebrow { color:#c8c5ff; font-size:11px; letter-spacing:2px; font-weight:700; text-transform:uppercase; }
    .hero h1 { font:800 clamp(27px,3.5vw,42px) Manrope,sans-serif; line-height:1.14; letter-spacing:-1.5px; margin:10px 0 12px; color:white; }
    .hero p { color:#e1e3f7; font-size:14px; max-width:650px; line-height:1.65; margin:0; }
    .hero-pill {
        display:inline-flex; align-items:center; gap:7px; margin-top:20px;
        border:1px solid rgba(255,255,255,.2); background:rgba(255,255,255,.09);
        border-radius:100px; padding:7px 12px; font-size:11px; color:#f1f0ff;
    }
    .section-heading { font:800 19px Manrope,sans-serif; letter-spacing:-.5px; color:#202943; margin:4px 0 4px; }
    .section-sub { font-size:12px; color:#7c8498; margin-bottom:14px; }
    .metric {
        background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:17px;
        padding:17px 18px; min-height:104px; box-shadow:0 4px 15px rgba(27,39,74,.025);
    }
    .metric-label { font-size:11px; color:#7b8498; font-weight:600; text-transform:uppercase; letter-spacing:.8px; }
    .metric-value { font:800 26px Manrope,sans-serif; color:#222b46; margin-top:7px; }
    .metric-note { font-size:11px; color:#8c94a7; margin-top:2px; }
    .paper-card {
        background:#fff; border:1px solid #e7eaf2; border-radius:16px; padding:17px 18px;
        margin:0 0 11px; box-shadow:0 5px 16px rgba(23,32,51,.025);
    }
    .paper-top { display:flex; justify-content:space-between; align-items:center; gap:12px; }
    .paper-number { color:#7a6cf4; background:#f0eeff; border-radius:9px; padding:6px 9px; font-size:11px; font-weight:800; white-space:nowrap; }
    .paper-title { font:700 15px Manrope,sans-serif; color:#202943; line-height:1.45; margin:10px 0 8px; }
    .paper-meta { font-size:12px; color:#7c8498; line-height:1.7; }
    .paper-link { color:#6457dc; font-weight:700; text-decoration:none; font-size:12px; }
    .tag {
        display:inline-block; padding:5px 9px; border-radius:8px; background:#eefaf5;
        color:#147b5a; font-size:10px; font-weight:700; margin-top:10px;
    }
    .result-panel {
        background:#fff; border:1px solid #e7eaf2; border-radius:18px; padding:22px;
        box-shadow:0 6px 22px rgba(23,32,51,.035); margin-bottom:15px;
    }
    .result-title { font:800 17px Manrope,sans-serif; color:#202943; margin-bottom:9px; }
    .result-body { font-size:13px; color:#515b72; line-height:1.8; }
    .question {
        border:1px solid #ecebff; background:linear-gradient(110deg,#faf9ff,#fff);
        border-radius:13px; padding:13px 15px; margin:8px 0; color:#333b56; font-size:13px; line-height:1.6;
    }
    .footnote { color:#8b93a6; font-size:11px; margin-top:20px; }
    div.stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#7667f5,#5a4be7); border:0; border-radius:12px;
        color:white; font-weight:700; padding:.7rem 1.1rem; box-shadow:0 7px 18px rgba(118,103,245,.22);
    }
    div.stButton > button[kind="primary"]:hover { border:0; background:linear-gradient(135deg,#6859ed,#4e40d8); }
    .stTextInput input, .stTextArea textarea { border-radius:12px !important; }
    div[data-testid="stExpander"] { border-radius:13px; border:1px solid #e7eaf2; background:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">🧭</div>
          <div><div class="brand-title">ResearchPilot</div>
          <div class="brand-sub">AI Research Workspace</div></div>
        </div>
        <div class="side-label">WORKSPACE</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Discover literature. Find direction. Build stronger research.")
    st.markdown('<div class="side-label">YOUR WORKFLOW</div>', unsafe_allow_html=True)
    st.markdown("①  Research topic")
    st.markdown("②  Scholarly discovery")
    st.markdown("③  AI research plan")
    st.markdown("④  Review & export")
    st.markdown(
        """
        <div class="side-card">
          <div class="side-card-title">CURRENT ENGINE</div>
          <div class="side-card-value">✦ Gemini + Crossref</div>
          <div class="side-card-title" style="margin-top:9px">MODE</div>
          <div class="side-card-value">Research planning MVP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="footnote">ResearchPilot · Week 2 MVP<br/>Built for evidence-led exploration.</div>', unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">✦ YOUR AI-POWERED RESEARCH DESK</div>
      <h1>Turn a research question<br/>into a clear direction.</h1>
      <p>Explore scholarly literature, surface promising papers, and generate a structured research plan — all from one focused workspace.</p>
      <div class="hero-pill">● Crossref literature search &nbsp; · &nbsp; Gemini plan generation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Search form ----------
st.markdown('<div class="section-heading">Start with a research topic</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Enter a focused topic to discover relevant scholarly records and build a first-pass research plan.</div>', unsafe_allow_html=True)

with st.form("research_form", clear_on_submit=False):
    topic_col, limit_col = st.columns([4, 1])
    with topic_col:
        topic = st.text_input(
            "Research topic",
            value=st.session_state.get("last_topic", "Machine learning for weather forecasting"),
            placeholder="e.g., Explainable AI for medical image analysis",
            label_visibility="collapsed",
        )
    with limit_col:
        limit = st.selectbox("Papers", [5, 10, 15], index=0)
    submitted = st.form_submit_button("✦  Discover papers & generate plan", type="primary", use_container_width=True)

# ---------- Main content ----------
if submitted:
    if not topic.strip():
        st.warning("Enter a research topic to continue.")
    elif not GEMINI_API_KEY:
        st.error("Gemini API key not found. Add GEMINI_API_KEY to your .env file and restart the app.")
    else:
        st.session_state["last_topic"] = topic.strip()
        progress = st.empty()
        try:
            progress.info("Searching Crossref for scholarly records…")
            papers = search_papers(topic.strip(), limit=limit)
            if not papers:
                progress.empty()
                st.warning("No records found. Try a broader or more specific topic.")
            else:
                st.session_state["papers"] = papers
                st.session_state["research_topic"] = topic.strip()
                st.session_state.pop("research_plan", None)
                progress.success(f"Retrieved {len(papers)} paper records. Preparing your research plan…")
                plan = create_research_plan(topic.strip(), papers)
                st.session_state["research_plan"] = plan
                st.session_state["generated_at"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
                progress.empty()
                st.success("Research plan generated successfully.")
        except Exception as exc:
            progress.empty()
            # Preserve any papers already retrieved so they remain visible if Gemini fails.
            st.error(f"Could not complete generation: {type(exc).__name__}: {exc}")
            st.caption("If papers were retrieved before the error, they remain available below. Gemini service errors may be temporary.")

papers = st.session_state.get("papers", [])
plan = st.session_state.get("research_plan")
active_topic = st.session_state.get("research_topic", topic if "topic" in locals() else "")

if papers:
    st.markdown("---")
    st.markdown('<div class="section-heading">Your discovery results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Showing scholarly metadata returned by Crossref for <b>{active_topic}</b>.</div>', unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metrics = [
        ("PAPERS RETRIEVED", str(len(papers)), "Crossref records"),
        ("PLAN STATUS", "Ready" if plan else "Pending", "Gemini generation"),
        ("RESEARCH QUESTIONS", str(len(plan.get("research_questions", []))) if plan else "—", "Structured questions"),
        ("SELECTED PAPERS", str(len(plan.get("papers", []))) if plan else "—", "Chosen for the plan"),
    ]
    for col, (label, value, note) in zip(metric_cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown('<div class="section-heading" style="margin-top:22px">Scholarly papers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Review source details before relying on a paper.</div>', unsafe_allow_html=True)
        for i, paper in enumerate(papers, start=1):
            authors = ", ".join(paper.get("authors", [])) or "Author information unavailable"
            year = paper.get("year") or "Year unavailable"
            doi = paper.get("doi") or "Not available"
            url = paper.get("url")
            link_html = f'<a class="paper-link" href="{url}" target="_blank" rel="noopener noreferrer">Open source ↗</a>' if url else ""
            st.markdown(
                f"""
                <div class="paper-card">
                  <div class="paper-top"><span class="paper-number">PAPER {i:02d}</span><span style="font-size:11px;color:#8b93a6">{year}</span></div>
                  <div class="paper-title">{paper.get("title", "Untitled record")}</div>
                  <div class="paper-meta"><b>Authors:</b> {authors}<br/><b>DOI:</b> {doi}</div>
                  <div style="margin-top:10px">{link_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-heading" style="margin-top:22px">AI research plan</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">A first-pass plan generated from the retrieved records.</div>', unsafe_allow_html=True)
        if not plan:
            st.info("Your papers are ready. Research plan generation did not finish. You can retry below.")
            if st.button("Retry plan generation", type="primary", use_container_width=True):
                try:
                    with st.spinner("Asking Gemini to generate the plan…"):
                        new_plan = create_research_plan(active_topic, papers)
                    st.session_state["research_plan"] = new_plan
                    st.rerun()
                except Exception as exc:
                    st.error(f"Retry failed: {type(exc).__name__}: {exc}")
        else:
            st.markdown(
                f'<div class="result-panel"><div class="result-title">Research snapshot</div><div class="result-body">{plan.get("summary", "No summary returned.")}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="result-panel"><div class="result-title">Research questions</div>', unsafe_allow_html=True)
            for i, question in enumerate(plan.get("research_questions", []), start=1):
                st.markdown(f'<div class="question"><b>Q{i:02d}</b> &nbsp; {question}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Research limitations & evidence gaps", expanded=True):
                st.write(plan.get("limitations", "No limitations provided."))

            with st.expander("Selected papers"):
                selected = plan.get("papers", [])
                if selected:
                    for paper in selected:
                        st.markdown(f"**{paper.get('title', 'Untitled')}**")
                        st.caption(f"Year: {paper.get('year') or 'Unknown'} · DOI: {paper.get('doi') or 'Unavailable'}")
                else:
                    st.write("No selected papers were returned.")

            export_payload = {
                "topic": active_topic,
                "generated_at": st.session_state.get("generated_at"),
                "research_plan": plan,
                "retrieved_papers": papers,
            }
            st.download_button(
                "↓  Export research plan (JSON)",
                data=json.dumps(export_payload, indent=2, ensure_ascii=False),
                file_name="researchpilot_plan.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
            )
            st.caption(f"Generated {st.session_state.get('generated_at', '')}")

    st.markdown('<div class="footnote">ResearchPilot is an early-stage research assistant. Crossref metadata and AI-generated interpretations should be verified against original papers before academic use.</div>', unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="result-panel" style="margin-top:24px">
          <div class="result-title">Your research workspace is ready</div>
          <div class="result-body">Your paper cards and research plan will appear here after you run a search. Start with a clear topic, inspect the sources, then review the generated questions and limitations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    empty_cols = st.columns(3)
    features = [
        ("01", "Discover literature", "Search scholarly metadata through Crossref."),
        ("02", "Shape the question", "Turn the topic into focused research questions."),
        ("03", "Export your plan", "Download the structured output as JSON."),
    ]
    for col, (num, title, body) in zip(empty_cols, features):
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">STEP {num}</div><div class="result-title" style="margin-top:10px">{title}</div><div class="result-body">{body}</div></div>',
                unsafe_allow_html=True,
            )
