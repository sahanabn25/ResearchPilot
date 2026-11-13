"""Deterministic evidence checks (no LLM). These produce the facts the evaluation rests on.

Why this exists: whether a result "supports" a claimed improvement, and whether a
technique appears in prior work, must not depend on a language model's opinion.
The LLM only structures text and writes qualitative judgement on top of these facts.
"""
import re
from typing import Optional

from .agents.paper_analysis import NOT_REPORTED, _supported, _token_set

HIGHER_IS_BETTER = {"accuracy", "acc", "f1", "f-score", "fscore", "precision", "recall", "auc", "auroc", "map",
                    "bleu", "rouge", "r2", "r-squared", "iou", "dice", "ndcg", "mrr", "specificity", "sensitivity",
                    "skill", "correlation"}
LOWER_IS_BETTER = {"rmse", "mae", "mse", "loss", "error", "latency", "perplexity", "wer", "cer", "mape",
                   "fid", "runtime", "rmsd"}

_NUM = re.compile(r"\d+(?:\.\d+)?")


def metric_direction(name: str) -> Optional[str]:
    """'higher' / 'lower' if the metric's better direction is well known, else None."""
    lowered = (name or "").lower()
    # both the whole hyphenated word ("r-squared") and its parts ("f1" in "f1-score") may be known metrics
    tokens = set(re.findall(r"[a-z0-9\-]+", lowered)) | set(re.findall(r"[a-z0-9]+", lowered))
    higher, lower = bool(tokens & HIGHER_IS_BETTER), bool(tokens & LOWER_IS_BETTER)
    if higher == lower:  # unknown, or contradictory (e.g. "error rate accuracy")
        return None
    return "higher" if higher else "lower"


def parse_number(text) -> Optional[float]:
    match = _NUM.search(str(text or ""))
    return float(match.group()) if match else None


def find_metric_value(text: str, metric: str) -> Optional[float]:
    """Number stated right after `metric` in `text`; None if absent or ambiguous."""
    if not text or not metric or text.strip().lower() == NOT_REPORTED:
        return None
    pattern = re.escape(metric.strip()) + r"[^0-9.\n]{0,20}?(\d+(?:\.\d+)?)"
    values = {float(m.group(1)) for m in re.finditer(pattern, text, flags=re.IGNORECASE)}
    return values.pop() if len(values) == 1 else None


def check_improvement(metric: str, proposed, baseline) -> dict:
    """Judge a claimed improvement using ONLY the numbers the user provided."""
    p, b = parse_number(proposed), parse_number(baseline)
    if p is None or b is None:
        return {"status": "not_verifiable",
                "reason": "a numeric value for the proposed method and for the baseline are both needed"}
    direction = metric_direction(metric)
    base = {"metric": metric, "proposed": p, "baseline": b, "direction": direction}
    if direction is None:
        return {**base, "status": "not_verifiable",
                "reason": "it is unclear whether a higher or lower value is better for this metric"}
    if p == b:
        return {**base, "status": "no_difference", "relative_change_pct": 0.0}
    better = p > b if direction == "higher" else p < b
    change = round(abs(p - b) / abs(b) * 100, 1) if b else None
    return {**base, "status": "numbers_favour_claim" if better else "numbers_contradict_claim",
            "relative_change_pct": change,
            "caveat": "Based on a single reported comparison; no significance testing is assumed."}


_RIGOR_PATTERNS = {
    "held_out_evaluation": r"test (?:set|split|data)|held[- ]?out|validation (?:set|split)|train/test|hold[- ]?out",
    "cross_validation": r"cross[- ]?validation|k-?fold|\b\d+-fold",
    "repeated_runs_or_variance": r"\bseeds?\b|multiple runs|repeated|averaged over|std\b|standard deviation|±",
    "statistical_test": r"p[- ]?value|t-test|wilcoxon|confidence interval|significan",
    "baseline_comparison": r"baseline|compared (?:with|to|against)|versus|\bvs\.?\b|state[- ]of[- ]the[- ]art",
    "ablation_study": r"ablation",
    "compute_cost_reported": r"latency|training time|inference time|\bgpu\b|runtime|flops|memory (?:usage|footprint)",
}


def rigor_check(user_text: str) -> dict:
    """Which standard experimental-rigor elements the user's answers mention (not whether they exist)."""
    text = user_text or ""
    return {k: bool(re.search(p, text, flags=re.IGNORECASE)) for k, p in _RIGOR_PATTERNS.items()}


def prior_use(items: list, analyses: dict, fields: tuple) -> list:
    """For each user item (technique/dataset), which analysed papers mention it (word-level match)."""
    ok = {pid: a for pid, a in analyses.items() if a.get("status") == "ok"}
    token_sets = {}
    for pid, a in ok.items():
        parts = []
        for f in fields:
            v = a.get(f)
            parts.extend(v if isinstance(v, list) else [str(v or "")])
        token_sets[pid] = _token_set(" ".join(parts))
    return [{"item": item, "paper_ids": [pid for pid, ts in token_sets.items() if _supported(item, ts)]}
            for item in items]


def numeric_comparisons(metrics: list, datasets: list, analyses: dict) -> dict:
    """Compare the user's metric values with values verified in retrieved papers.

    Only comparable when the paper used the same dataset AND states a single value for
    the same metric. Everything else is reported as not comparable, never guessed.
    """
    rows, considered = [], 0
    for m in metrics:
        user_value = parse_number(m.get("value"))
        if user_value is None:
            continue
        for pid, a in analyses.items():
            if a.get("status") != "ok":
                continue
            considered += 1
            ds_tokens = _token_set(" ".join(a.get("datasets") or []))
            same_dataset = any(_supported(d, ds_tokens) for d in datasets) if datasets and ds_tokens else False
            paper_value = find_metric_value(a.get("key_results", ""), m["name"])
            if not (same_dataset and paper_value is not None):
                continue
            direction = metric_direction(m["name"])
            better = None if direction is None else (user_value > paper_value if direction == "higher"
                                                     else user_value < paper_value)
            rows.append({"metric": m["name"], "user_value": user_value, "paper_id": pid, "title": a.get("title", ""),
                         "paper_value": paper_value, "direction": direction, "user_value_better": better,
                         "provenance": "verified_from_papers"})
    return {"comparisons": rows, "pairs_considered": considered,
            "note": ("Only same-dataset, same-metric pairs with a single stated value are compared. "
                     "Differences in splits, preprocessing or settings are not controlled for.")}
