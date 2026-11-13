"""Build the structured research analysis report (14 sections) from a WorkflowState.

This module uses NO language model. Everything in the report is read from the state (retrieved paper
metadata, verified analyses, gaps, the user's answers, computed evidence), so nothing can be invented here.
The optional AI narrative (see agents/report_writer.py) is passed in separately, already validated, and
is always labelled as AI-generated.

Report format (JSON-serialisable dict):
  {"title", "subtitle", "generated_at", "mode", "topic", "complete": bool, "notices": [str],
   "sections": [{"number", "title", "blocks": [block]}], "references": [{"n", "paper_id", "text"}]}
Blocks: paragraph {text}, bullets {items}, table {header, rows, widths?, caption?}, note {text},
        subheading {text}. Each may carry "provenance": verified_from_papers | user_provided | ai_generated | None.
"""
import re
from collections import Counter
from datetime import datetime
from typing import Optional

from ..agents.paper_analysis import NOT_REPORTED, has_usable_abstract
from .text import one_line

V, U, A = "verified_from_papers", "user_provided", "ai_generated"
LABELS_UPLOAD = {V: "Verified from related papers", U: "From uploaded paper", A: "AI-generated"}

SECTION_TITLES = [
    "Project Overview", "Research Problem", "Existing Research Analysis", "Literature Review Summary",
    "Identified Research Gaps", "User's Proposed Methodology", "Methodology Comparison",
    "Experimental Results and Performance Analysis", "Comparison with Existing Research",
    "Research Contributions", "Limitations", "Recommendations for Improvement", "Conclusion", "References"]

CHECK_TEXT = {"numbers_favour_claim": "Reported numbers favour the claim",
              "numbers_contradict_claim": "Reported numbers contradict the claim",
              "no_difference": "No difference in the reported numbers",
              "not_verifiable": "Not verifiable from the numbers provided"}
ADDRESSED_TEXT = {"yes": "Yes", "partially": "Partly", "no": "No", "unclear": "Unclear"}
METHOD_TEXT = {"different": "Different from approaches in the retrieved papers",
               "similar": "Similar to approaches in the retrieved papers", "unclear": "Unclear"}
RIGOR_LABEL = {"held_out_evaluation": "Held-out test or validation set", "cross_validation": "Cross-validation",
               "repeated_runs_or_variance": "Repeated runs or variance reported",
               "statistical_test": "Statistical significance test", "baseline_comparison": "Comparison with a baseline",
               "ablation_study": "Ablation study", "compute_cost_reported": "Compute cost or run time reported"}
RIGOR_ADVICE = {
    "held_out_evaluation": "State how the data was split and report results on a held-out test set.",
    "cross_validation": "Consider cross-validation to show results do not depend on one split.",
    "repeated_runs_or_variance": "Repeat experiments with different random seeds and report the mean and spread.",
    "statistical_test": "Add a significance test or confidence intervals for the claimed improvements.",
    "baseline_comparison": "Compare against at least one established baseline on the same data.",
    "ablation_study": "Add an ablation study showing which components contribute to the result.",
    "compute_cost_reported": "Report training and inference cost so efficiency claims can be judged."}
ANALYSIS_LIMITS = [
    "Papers were analysed from their abstracts only; details that appear only in full texts are not captured.",
    "Technique and dataset matching is word-based, so synonyms and acronyms (for example CNN versus "
    "convolutional neural network) can be missed. 'Not found in retrieved papers' does not prove novelty.",
    "Comparisons are made against a limited set of retrieved papers, not the whole literature, and do not control "
    "for different data splits, preprocessing or experimental settings.",
    "AI-generated items are hypotheses to check, not findings. Verify everything against the original papers "
    "before using it academically."]


def P(text, prov=None):
    return {"type": "paragraph", "text": one_line(text, 1500), "provenance": prov}


def B(items, prov=None):
    return {"type": "bullets", "items": [one_line(i, 600) for i in items if i], "provenance": prov}


def T(header, rows, prov=None, caption="", widths=None):
    return {"type": "table", "header": [one_line(h, 60) for h in header],
            "rows": [[one_line(c, 260) for c in r] for r in rows], "provenance": prov,
            "caption": one_line(caption, 300), "widths": widths}


def N(text):
    return {"type": "note", "text": one_line(text, 900), "provenance": None}


def H(text):
    return {"type": "subheading", "text": one_line(text, 120), "provenance": None}


def num(value) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


class _Ctx:
    """Everything the section builders need, resolved once."""

    def __init__(self, state, narrative, generated_at):
        self.state, self.narrative = state, narrative or {}
        self.generated_at = generated_at
        self.analyses = state.analyses or {}
        self.ref_map = reference_map(state)
        self.papers = {p.get("paper_id"): p for p in state.papers}
        self.fields = (state.profile or {}).get("fields", {})
        self.gap_responses = (state.profile or {}).get("gap_responses", [])
        self.ev = state.evaluation or {}
        self.claims = self.ev.get("claims") or {}
        self.evidence = self.ev.get("evidence") or {}
        self.assess = self.ev.get("assessment") or {}
        self.gaps = state.gaps or {}
        self.upload = state.mode == "upload"
        self.paper = state.paper or {}
        self.parsed = self.paper.get("parsed") or {}

    def cite(self, ids) -> str:
        nums = sorted({self.ref_map[i] for i in (ids or []) if i in self.ref_map})
        return "[" + ", ".join(map(str, nums)) + "]" if nums else ""

    def ok_analyses(self):
        return [(pid, self.analyses[pid]) for pid in self.ref_map]


def reference_map(state) -> dict:
    """paper_id -> reference number, for successfully analysed papers (in analysis order)."""
    mapping = {}
    for pid, a in (state.analyses or {}).items():
        if a.get("status") == "ok":
            mapping[pid] = len(mapping) + 1
    return mapping


def format_reference(paper: dict, analysis: dict) -> str:
    authors = [a for a in (paper.get("authors") or []) if a]
    who = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "") if authors else "Author information unavailable"
    year = paper.get("year") or analysis.get("year") or "n.d."
    title = paper.get("title") or analysis.get("title") or "Untitled"
    parts = [f"{who} ({year}). {title}."]
    if paper.get("venue"):
        parts.append(f"{paper['venue']}.")
    if paper.get("doi"):
        parts.append(f"DOI: {paper['doi']}")
    elif paper.get("url"):
        parts.append(str(paper["url"]))
    return one_line(" ".join(parts), 500)


# ------------------------------------------------------------------ sections
def _s1_overview(c: _Ctx):
    st = c.state
    n_abs = sum(1 for p in st.papers if has_usable_abstract(p))
    sources = ", ".join(f"{k}: {v}" for k, v in (st.source_status or {}).items()) or "not recorded"
    rows = [["Report type", "Topic-based literature analysis" if st.mode == "topic" else "Uploaded-paper analysis"],
            ["Search topic (the paper's title)" if c.upload else "Research topic", st.topic or "not provided"],
            ["Paper title" if c.upload else "Project title (as provided)", c.fields.get("title") or "not provided"]]
    if c.upload:
        pp = c.paper
        rows += [["Uploaded file", pp.get("filename", "not recorded")],
                 ["Pages read", f"{pp.get('pages_read', '?')} of {pp.get('n_pages', '?')}"],
                 ["Sections detected", ", ".join(pp.get("section_order", [])) or "none"],
                 ["Tables, equations, figures detected",
                  f"{len(pp.get('tables', []))}, {len(pp.get('equations', []))}, {len(pp.get('figures', []))}"]]
    rows += [["Generated", c.generated_at], ["Sources searched", sources],
             ["Related papers retrieved" if c.upload else "Papers retrieved", str(len(st.papers))],
             ["With usable abstracts", str(n_abs)], ["Papers analysed", str(len(c.ref_map))],
             ["Workflow steps used", str(st.steps_used)]]
    purpose = ("This report analyses the uploaded research paper, compares it with related published work found by "
               "searching the scholarly databases, and identifies gaps. Each statement is labelled by where it comes from."
               if c.upload else
               "This report summarises the research landscape for the topic above, identifies gaps in the retrieved "
               "literature, and compares the user's project with it. Each statement is labelled by where it comes from.")
    legend = ["Verified from related papers: derived by code from the related papers' own abstracts.",
              "From uploaded paper: stated by the paper's authors; not independently verified."
              if c.upload else "User-provided: stated by the user and not independently verified.",
              "AI-generated: model judgement or suggestion; treat it as a hypothesis, not a finding."]
    if not c.upload:
        legend[0] = "Verified from papers: derived by code from the retrieved papers' own abstracts."
    blocks = [P(purpose), T(["Item", "Value"], rows, widths=[1.9, 4.1]), H("How to read the labels"), B(legend)]
    notes = [w for w in c.paper.get("warnings", []) if not w.startswith("Equations are recovered")] if c.upload else []
    if notes:
        blocks.append(N("Extraction notes: " + " ".join(notes)))
    return blocks


def _s2_problem(c: _Ctx):
    blocks = []
    if c.fields.get("problem"):
        blocks += [H("Problem addressed by the paper" if c.upload else "Problem addressed by the user's project"),
                   P(c.fields["problem"], U)]
    else:
        blocks.append(N("The paper's research problem could not be extracted." if c.upload
                        else "The user's research problem was not provided."))
    if c.fields.get("objectives"):
        blocks += [H("Objectives"), P(c.fields["objectives"], U)]
    grouped = {}                                   # identical statements are merged: "text [1, 2, 3]"
    for pid, a in c.ok_analyses():
        text = (a.get("research_problem") or "").strip()
        if text and text.lower() != NOT_REPORTED:
            grouped.setdefault(text, []).append(pid)
    framed = [f"{text} {c.cite(ids)}" for text, ids in list(grouped.items())[:6]]
    blocks.append(H("How the retrieved papers frame the problem"))
    blocks.append(B(framed, V) if framed else N("No research problem statements were reported in the analysed abstracts."))
    return blocks


def _s3_existing(c: _Ctx):
    rows = []
    for pid, a in c.ok_analyses():
        rows.append([f"[{c.ref_map[pid]}]", f"{a.get('title', '')} ({a.get('year') or 'n.d.'})", a.get("methodology", ""),
                     ", ".join(a.get("techniques") or []) or NOT_REPORTED, ", ".join(a.get("datasets") or []) or NOT_REPORTED,
                     a.get("key_results", "")])
    if not rows:
        return [N("No papers could be analysed, so there is no existing-research analysis to show.")]
    s = c.state.analysis_stats or {}
    parts = [(s.get("skipped_no_abstract", 0), "without a usable abstract"),
             (s.get("not_selected", 0), "beyond the analysis limit"), (s.get("failed", 0), "failed")]
    parts = [f"{n} {label}" for n, label in parts if n]
    skipped = ("Not analysed: " + ", ".join(parts) + ".") if parts else ""
    return [T(["Ref", "Paper (year)", "Methodology", "Techniques", "Datasets", "Key results"], rows, V,
              "Abstract-level extraction. Named items and numbers were checked against each abstract's own text.",
              widths=[0.4, 1.7, 1.4, 1.2, 0.9, 1.4])] + ([N(skipped)] if skipped else [])


def _techniques_counter(c: _Ctx, field):
    counter = Counter()
    for _, a in c.ok_analyses():
        for item in {i.strip().lower() for i in a.get(field) or [] if i.strip()}:
            counter[item] += 1
    return counter.most_common(5)


def _s4_literature(c: _Ctx):
    blocks = []
    paragraphs = c.narrative.get("literature_review") or []
    if paragraphs:
        blocks += [P(t, A) for t in paragraphs]
    else:
        years = [a.get("year") for _, a in c.ok_analyses() if isinstance(a.get("year"), int)]
        span = f", published between {min(years)} and {max(years)}" if years else ""
        techs, datas = _techniques_counter(c, "techniques"), _techniques_counter(c, "datasets")
        text = f"{len(c.ref_map)} papers with usable abstracts were analysed{span}."
        if techs:
            text += " Most frequently reported techniques: " + "; ".join(f"{t} ({n})" for t, n in techs) + "."
        if datas:
            text += " Most frequently reported datasets: " + "; ".join(f"{d} ({n})" for d, n in datas) + "."
        blocks.append(P(text, V) if c.ref_map else N("No analysed papers are available for a literature summary."))
    trends = [f"{t['trend']} {c.cite(t['paper_ids'])} (supported by {len(t['paper_ids'])} paper(s); "
              f"{t.get('evidence_strength', 'weak')})" for t in c.gaps.get("trends", [])]
    if trends:
        blocks += [H("Trends across the papers"), B(trends, A)]
    lims = [f"{t['limitation']} {c.cite(t['paper_ids'])}" for t in c.gaps.get("common_limitations", [])]
    if lims:
        blocks += [H("Limitations common to the papers"), B(lims, A)]
    return blocks


def _s5_gaps(c: _Ctx):
    gaps = c.gaps.get("gaps", [])
    if not c.gaps:
        return [N("Gap identification has not been run.")]
    if c.gaps.get("insufficient_evidence") or not gaps:
        return [N(c.gaps.get("message") or "No research gaps were identified from the retrieved papers.")]
    rows = [[str(i), g["gap"], g.get("evidence_strength", "weak"), c.cite(g["paper_ids"]), g.get("rationale", "")]
            for i, g in enumerate(gaps, 1)]
    return [T(["#", "Research gap", "Evidence strength", "Supporting papers", "Rationale"], rows, A,
              "Gap wording is AI-generated. Supporting papers and evidence strength are computed by code.",
              widths=[0.3, 2.2, 0.9, 0.9, 1.9]), N(c.gaps.get("scope_note", ""))]


def _s6_method(c: _Ctx):
    if not c.fields:
        return [N("The paper's methodology could not be extracted." if c.upload else "The user's methodology was not provided.")]
    labels = [("title", "Title"), ("objectives", "Objectives"), ("methodology", "Methodology"), ("techniques", "Techniques"),
              ("datasets", "Datasets"), ("advantages", "Claimed contributions" if c.upload else "Stated advantages"),
              ("difference", "Stated difference from existing work")]
    rows = [[lab, c.fields[k]] for k, lab in labels if c.fields.get(k)]
    blocks = [T(["Item", "As stated in the uploaded paper" if c.upload else "As provided by the user"], rows, U,
                widths=[1.6, 4.4])]
    if c.claims:
        verified = [f"Technique: {t}" for t in c.claims.get("techniques", [])] + \
                   [f"Dataset: {d}" for d in c.claims.get("datasets", [])]
        if verified:
            blocks += [H("Items confirmed in the paper's text" if c.upload else "Items confirmed in the user's own answers"),
                       B(verified, U)]
        dropped = c.claims.get("dropped") or {}
        count = sum(len(v) for v in dropped.values())
        if count:
            # Only the COUNT is shown: repeating the rejected values would put unverified content in the report.
            where = "the paper" if c.upload else "the answers"
            blocks.append(N(f"{count} item(s) extracted from {where} (techniques, datasets or numbers) were removed because "
                            "they could not be found in the " + ("paper's" if c.upload else "user's own") + " text."))
    if c.upload and c.paper.get("equations"):
        blocks += [H("Equation-like lines detected in the paper"), B(c.paper["equations"][:8], U),
                   N("Recovered from the PDF text; symbols and layout may be lost. Equations are listed, not interpreted.")]
    return blocks


def _s7_method_comparison(c: _Ctx):
    rows = []
    for kind, key in (("Technique", "technique_prior_use"), ("Dataset", "dataset_overlap")):
        for r in c.evidence.get(key, []):
            found = c.cite(r.get("paper_ids"))
            rows.append([r.get("item", ""), kind, found + " mentioned" if found else "Not found by word matching"])
    blocks = []
    if rows:
        blocks.append(T(["User's item", "Type", "In the retrieved papers"], rows, V,
                        "Word-level matching against the analysed abstracts.", widths=[2.4, 1.0, 2.6]))
    else:
        blocks.append(N("No techniques or datasets from the user's project were available to compare."))
    md = c.assess.get("methodology_difference")
    if md:
        text = f"{METHOD_TEXT.get(md.get('assessment'), 'Unclear')}. {md.get('rationale', '')} {c.cite(md.get('similar_paper_ids'))}"
        blocks += [H("Assessment of methodological difference"), P(text, A)]
    if c.ev.get("scope_note"):
        blocks.append(N(c.ev["scope_note"]))
    return blocks


def _s8_results(c: _Ctx):
    blocks = []
    if c.fields.get("results"):
        blocks += [H("Results as stated in the paper" if c.upload else "Results as described by the user"),
                   P(c.fields["results"], U)]
    metrics = c.claims.get("metrics") or []
    if metrics:
        rows = [[m["name"], m["value"] or "not stated as a number", m.get("dataset") or "not stated"] for m in metrics]
        blocks += [H("Reported metrics"), T(["Metric", "Value", "Dataset"], rows, U, widths=[2.0, 2.0, 2.0])]
    checks = c.evidence.get("improvement_checks") or []
    if checks:
        rows = []
        for item in checks:
            chk = item["check"]
            change = f"{num(chk['relative_change_pct'])}%" if chk.get("relative_change_pct") is not None else "n/a"
            base = " ".join(x for x in [item.get("baseline_name"), item.get("baseline_value")] if x) or "not provided"
            rows.append([item["claim"], item.get("metric") or "n/a", item.get("proposed_value") or "n/a", base,
                         CHECK_TEXT.get(chk["status"], chk["status"]), change])
        blocks += [H("Do the reported numbers support the claimed improvements?"),
                   T(["Claim", "Metric", "Proposed", "Baseline", "Result of check", "Relative change"], rows, U,
                     "Numbers are the user's own; the check was computed by code. Based on a single reported "
                     "comparison, with no significance testing assumed.", widths=[1.9, 0.8, 0.8, 1.0, 1.4, 0.8])]
    rigor = c.evidence.get("rigor")
    if rigor:
        rows = [[RIGOR_LABEL.get(k, k), "Mentioned" if v else "Not mentioned"] for k, v in rigor.items()]
        blocks += [H("Experimental rigor mentioned in the paper" if c.upload else "Experimental rigor mentioned in the user's answers"),
                   T(["Element", "In the paper" if c.upload else "In the user's answers"], rows, U,
                     c.evidence.get("rigor_note", ""), widths=[3.5, 2.5])]
    if c.upload:
        tables = c.paper.get("tables") or []
        if tables:
            rows = [[t.get("caption") or f"Table on page {t['page']}", str(t["page"]),
                     f"{len(t['rows'])} rows x {max(len(r) for r in t['rows'])} columns" if t.get("rows") else "caption only"]
                    for t in tables[:12]]
            blocks += [H("Tables detected in the paper"),
                       T(["Table", "Page", "Extracted"], rows, U, "Table contents were read by the extraction step "
                         "and used when finding metrics; they are not reproduced here.", widths=[3.6, 0.6, 1.8])]
        figures = c.paper.get("figures") or []
        if figures:
            blocks += [H("Figures (captions only)"), B([f"{f['label']}: {f['caption']}" for f in figures[:10]], U),
                       N("The data inside graphs is not read; only captions are listed.")]
    return blocks or [N("No results or performance information was provided.")]


def _bool_text(better, user_value, paper_value):
    if user_value == paper_value:
        return "Equal"
    return {True: "User's value is better", False: "Paper's value is better"}.get(better, "Direction unknown")


def _s9_comparison(c: _Ctx):
    blocks = []
    nc = c.evidence.get("numeric_comparisons") or {}
    rows = [[r["metric"], num(r["user_value"]), num(r["paper_value"]),
             f"{c.cite([r['paper_id']])} {r.get('title', '')}", _bool_text(r.get("user_value_better"), r["user_value"], r["paper_value"])]
            for r in nc.get("comparisons", [])]
    if rows:
        blocks += [H("Same-dataset numeric comparisons"),
                   T(["Metric", "User's value", "Paper's value", "Paper", "Comparison"], rows, V, nc.get("note", ""),
                     widths=[0.9, 0.9, 0.9, 2.4, 1.2])]
    else:
        blocks.append(N("No retrieved paper reported a comparable value for the same dataset and metric, so no "
                        "numeric comparison is possible."))
    align = c.assess.get("gap_alignment") or []
    if align:
        rows = [[g["gap"], ADDRESSED_TEXT.get(g["addressed"], g["addressed"]), g.get("rationale", ""), c.cite(g.get("paper_ids"))]
                for g in align]
        blocks += [H("Does the project address the identified gaps?"),
                   T(["Research gap", "Addressed?", "Rationale", "Papers"], rows, A, widths=[2.2, 0.8, 2.2, 0.8])]
    elif c.fields and not c.assess:
        blocks.append(N("The qualitative assessment was not available, so gap alignment is not shown."))
    return blocks


def _s10_contributions(c: _Ctx):
    blocks, claimed = [], c.claims.get("claimed_contributions") or []
    if claimed:
        blocks += [H("Contributions claimed by the user"), B(claimed, U)]
    if c.fields.get("advantages"):
        blocks += [H("Advantages stated by the user"), P(c.fields["advantages"], U)]
    strengths = [i["strength"] + (f" (basis: {i['basis']})" if i.get("basis") else "")
                 for i in c.assess.get("strengths") or []]
    if strengths:
        blocks += [H("Strengths (AI assessment)"), B(strengths, A)]
    potential = []
    for item in c.assess.get("potential_contributions") or []:
        text = item["contribution"] + (f" (basis: {item['basis']})" if item.get("basis") else "")
        if item.get("caveat"):
            text += f" CAVEAT: {item['caveat']}"
        potential.append(text)
    if potential:
        blocks += [H("Potential contributions (AI assessment)"), B(potential, A)]
    return blocks or [N("No contributions were provided or could be assessed.")]


def _s11_limitations(c: _Ctx):
    blocks = []
    user_lims = list(c.claims.get("stated_limitations") or [])
    if not user_lims and c.fields.get("limitations"):
        user_lims = [c.fields["limitations"]]
    if user_lims:
        blocks += [H("Limitations stated by the authors" if c.upload else "Limitations stated by the user"), B(user_lims, U)]
    remaining = [i["limitation"] for i in c.assess.get("remaining_limitations") or [] if i.get("provenance") != U]
    if remaining:
        blocks += [H("Further limitations suggested by the assessment"), B(remaining, A)]
    blocks += [H("Limitations of this analysis"), B(ANALYSIS_LIMITS)]
    return blocks


def _plain_gap(text: str) -> str:
    """'Not addressed in the supplied papers: X' -> 'X' (gap statements often carry that prefix)."""
    return re.sub(r"^\s*not addressed( in the (supplied|retrieved) papers)?\s*[:\-]\s*", "", text, flags=re.IGNORECASE) or text


def _s12_recommendations(c: _Ctx):
    blocks = []
    experiments = [f"{e['experiment']} ({e['reason']})" if e.get("reason") else e["experiment"]
                   for e in c.assess.get("suggested_experiments") or []]
    if experiments:
        blocks += [H("Suggested additional experiments"), B(experiments, A)]
    open_gaps = [f"Consider whether the work could address: {_plain_gap(g['gap'])}"
                 for g in c.assess.get("gap_alignment") or [] if g.get("addressed") in ("no", "unclear")]
    if open_gaps:
        blocks += [H("Gaps not yet addressed"), B(open_gaps, A)]
    future = c.parsed.get("future_work") or []
    if c.upload and future:
        blocks += [H("Future directions stated by the authors"), B(future, U)]
    rigor = c.evidence.get("rigor") or {}
    advice = [RIGOR_ADVICE[k] for k, mentioned in rigor.items() if not mentioned and k in RIGOR_ADVICE]
    if advice:
        blocks += [H("Strengthening the evidence"),
                   N("Computed from what the user's answers mention. If you already did these, add the details."),
                   B(advice)]
    return blocks or [N("No recommendations could be generated because the paper was not evaluated." if c.upload
                        else "No recommendations could be generated because the user's project was not evaluated.")]


def _s13_conclusion(c: _Ctx):
    blocks = []
    paragraphs = c.narrative.get("conclusion") or []
    findings = [f"{len(c.ref_map)} papers on this topic were analysed from their abstracts.",
                f"{len(c.gaps.get('gaps', []))} candidate research gap(s) were identified."]
    align = c.assess.get("gap_alignment") or []
    if align:
        addressed = sum(1 for g in align if g.get("addressed") in ("yes", "partially"))
        findings.append(f"The project was assessed as addressing or partly addressing {addressed} of {len(align)} gap(s).")
    checks = c.evidence.get("improvement_checks") or []
    if checks:
        supported = sum(1 for i in checks if i["check"]["status"] == "numbers_favour_claim")
        findings.append(f"{supported} of {len(checks)} claimed improvement(s) are supported by the numbers the user provided.")
    nc = (c.evidence.get("numeric_comparisons") or {}).get("comparisons", [])
    findings.append(f"{len(nc)} same-dataset numeric comparison(s) with retrieved papers were possible.")
    if paragraphs:
        blocks += [P(t, A) for t in paragraphs]
    else:
        blocks.append(P("This report analysed the retrieved literature, identified candidate research gaps and, where the "
                        "user described their project, compared it with that literature. The findings are abstract-level "
                        "and should be verified against the full papers."))
    blocks += [H("Key findings"), B(findings), N("Counts above are computed from this analysis.")]
    return blocks


def _s14_references(c: _Ctx):
    blocks = []
    if c.upload:
        blocks.append(N(f"Paper analysed: {c.fields.get('title') or c.state.topic} ({c.paper.get('filename', 'uploaded file')}). "
                        "It is the subject of this report and is not numbered below."))
    refs = build_references(c.state)
    if refs:
        blocks.append({"type": "references", "items": [f"[{r['n']}] {r['text']}" for r in refs], "provenance": V})
    else:
        blocks.append(N("No papers were analysed, so there are no references."))
    others = [p for p in c.state.papers if p.get("paper_id") not in c.ref_map]
    if others:
        blocks += [H("Other retrieved records (not analysed, not cited above)"),
                   B([f"{p.get('title', 'Untitled')} ({p.get('year') or 'n.d.'})" for p in others[:15]])]
    blocks.append(N("Reference metadata comes from Crossref, OpenAlex and arXiv. Verify details before citing."))
    return blocks


def build_references(state) -> list:
    papers = {p.get("paper_id"): p for p in state.papers}
    return [{"n": n, "paper_id": pid, "text": format_reference(papers.get(pid, {}), state.analyses[pid])}
            for pid, n in reference_map(state).items()]


_BUILDERS = [_s1_overview, _s2_problem, _s3_existing, _s4_literature, _s5_gaps, _s6_method, _s7_method_comparison,
             _s8_results, _s9_comparison, _s10_contributions, _s11_limitations, _s12_recommendations, _s13_conclusion,
             _s14_references]


def build_report(state, narrative: Optional[dict] = None, generated_at: Optional[str] = None) -> dict:
    """Assemble the 14-section report. `narrative` is optional, pre-validated AI text (see report_writer)."""
    if not (state.topic or state.papers or state.analyses):
        raise ValueError("There is nothing to report yet. Run the topic analysis first.")
    now = datetime.now()
    generated_at = generated_at or f"{now.day} {now.strftime('%B %Y')}"
    c = _Ctx(state, narrative, generated_at)

    notices = []
    if not c.ref_map:
        notices.append("No papers could be analysed, so most sections are empty.")
    if not c.gaps or c.gaps.get("insufficient_evidence"):
        notices.append("Gap identification had too little evidence; no research gaps are listed.")
    if c.upload:
        if not state.profile:
            notices.append("The paper's content could not be extracted, so sections 2 and 6 to 12 show only what is missing.")
        elif not state.evaluation:
            notices.append("The evaluation of the uploaded paper did not complete.")
        if c.parsed.get("source") and c.parsed["source"] != "llm":
            notices += c.parsed.get("notes", [])[:2] or ["AI extraction was only partly available for this paper."]
    elif not state.profile:
        notices.append("The user's project was not described, so sections 6 to 12 show only what is missing.")
    elif not state.evaluation:
        notices.append("The user's answers were collected but the evaluation did not complete.")
    elif not c.assess:
        notices.append("The qualitative assessment was unavailable; only computed evidence is shown.")

    title_source = c.fields.get("title") or state.topic or "Untitled project"
    return {
        "title": "Research Analysis Report: " + one_line(title_source, 110),
        "subtitle": f"Generated by ResearchPilot on {generated_at}. AI-assisted: verify against the original papers.",
        "generated_at": generated_at, "mode": state.mode, "topic": state.topic,
        "labels": dict(LABELS_UPLOAD) if c.upload else None,
        "complete": not notices, "notices": notices,
        "sections": [{"number": i, "title": SECTION_TITLES[i - 1], "blocks": fn(c)}
                     for i, fn in enumerate(_BUILDERS, 1)],
        "references": build_references(state),
    }


# ------------------------------------------------------------------ checks and helpers
def validate_report(report: dict) -> None:
    """Structural validation. Raises ValueError describing the first problem."""
    sections = report.get("sections") or []
    if [s.get("title") for s in sections] != SECTION_TITLES:
        raise ValueError("The report does not contain the 14 required sections in order.")
    for s in sections:
        if not s.get("blocks"):
            raise ValueError(f"Section {s['number']} ({s['title']}) is empty.")
        for b in s["blocks"]:
            if b["type"] == "table":
                if not b["rows"]:
                    raise ValueError(f"Section {s['number']} has a table with no rows.")
                if any(len(r) != len(b["header"]) for r in b["rows"]):
                    raise ValueError(f"Section {s['number']} has a table with inconsistent columns.")
            elif b["type"] in ("bullets", "references") and not b["items"]:
                raise ValueError(f"Section {s['number']} has an empty list.")
            elif b["type"] in ("paragraph", "note", "subheading") and not b["text"].strip():
                raise ValueError(f"Section {s['number']} has an empty text block.")


def iter_text(report: dict):
    """Every text string in the report body (used for citation and number checks)."""
    for s in report["sections"]:
        for b in s["blocks"]:
            if b["type"] in ("paragraph", "note", "subheading"):
                yield b["text"]
            elif b["type"] in ("bullets", "references"):
                yield from b["items"]
            elif b["type"] == "table":
                yield b.get("caption", "")
                for row in b["rows"]:
                    yield from row


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def citation_problems(report: dict) -> list:
    """Citations like [3] that point to a reference that does not exist."""
    limit, problems = len(report.get("references", [])), []
    for text in iter_text(report):
        for group in _CITE.findall(text):
            for n in (int(x) for x in group.split(",")):
                if not 1 <= n <= limit:
                    problems.append(f"citation [{n}] has no matching reference")
    return problems
