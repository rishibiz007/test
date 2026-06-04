#!/usr/bin/env python3
"""Two evals for the Phoenix PM agent's report.

A. citation_groundedness — every #NNN in the report must exist in the input
   raw signal (phoenix_raw.json). Deterministic, no LLM needed.

B. priority_calibration  — every P0–P3 row in the Prioritized Backlog table is
   judged by Claude Haiku 4.5 against the actual cited issues' metadata.

Run from the taste_agent/ directory:
    python3 evals/pm_report_evals.py

If ARIZE_API_KEY + ARIZE_SPACE_ID are set, each eval is logged as an
OpenInference EVALUATOR span to Arize AX project `phoenix-pm-evals` so the
eval runs sit alongside the agent runs.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# allow `from instrumentation import ...` when run from evals/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instrumentation import setup_tracing, shutdown_tracing  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "phoenix_raw.json"
REPORT_PATH = REPO_ROOT / "phoenix_pm_report.md"

JUDGE_MODEL = "claude-haiku-4-5-20251001"
CITATION_RE = re.compile(r"#(\d{2,6})")

TOP_K = 20

REQUIRED_SECTIONS = [
    "Executive Summary",
    "Recent Release Context",
    "Top Pain Points",
    "Top Feature Asks",
    "Cross-Cutting Themes",
    "Prioritized Backlog",
    "Notable Single-Asker Asks",
]

# For eval E: which report section each citation lives under.
PAIN_POINTS_SECTION = "Top Pain Points"
FEATURE_ASKS_SECTION = "Top Feature Asks"

# Matches rows in the Prioritized Backlog markdown table.
# Captures: priority tag, theme, summary, refs cell.
BACKLOG_ROW_RE = re.compile(
    r"^\|\s*\*\*(P[0-3])\*\*\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw() -> dict[str, Any]:
    if not RAW_PATH.exists():
        sys.exit(f"ERROR: {RAW_PATH} not found — run phoenix_pm_agent.py first.")
    return json.loads(RAW_PATH.read_text())


def load_report() -> str:
    if not REPORT_PATH.exists():
        sys.exit(f"ERROR: {REPORT_PATH} not found — run phoenix_pm_agent.py first.")
    return REPORT_PATH.read_text()


def build_known_numbers(raw: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map issue/discussion number -> {kind, title, ...} for fast lookup."""
    known: dict[int, dict[str, Any]] = {}
    for item in raw.get("issues", []):
        known[item["number"]] = {**item, "source": "issue"}
    for item in raw.get("discussions", []):
        # discussions and issues share number space — issue wins if collision
        known.setdefault(item["number"], {**item, "source": "discussion"})
    return known


# ---------------------------------------------------------------------------
# Section splitter (shared by D + E)
# ---------------------------------------------------------------------------

def split_by_h2(report: str) -> dict[str, str]:
    """Return {section_title: section_body} for every H2 in the report."""
    sections: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []
    for line in report.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines)
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections[current_title] = "\n".join(current_lines)
    return sections


def find_section(sections: dict[str, str], needle: str) -> str | None:
    """Find the first section whose title starts with `needle` (case-insensitive)."""
    needle_l = needle.lower()
    for title, body in sections.items():
        if title.lower().startswith(needle_l):
            return body
    return None


# ---------------------------------------------------------------------------
# Eval A: citation groundedness
# ---------------------------------------------------------------------------

def eval_citation_groundedness(report: str, known: dict[int, dict[str, Any]]) -> dict[str, Any]:
    citations = sorted(set(int(m.group(1)) for m in CITATION_RE.finditer(report)))
    grounded = [n for n in citations if n in known]
    hallucinated = [n for n in citations if n not in known]
    total = len(citations)
    score = round(len(grounded) / total, 4) if total else 1.0

    return {
        "name": "citation_groundedness",
        "score": score,
        "passed": score == 1.0,
        "total_citations": total,
        "grounded": len(grounded),
        "hallucinated_count": len(hallucinated),
        "hallucinated_numbers": hallucinated,
        "explanation": (
            f"{len(grounded)}/{total} cited issue/discussion numbers exist in "
            f"the input payload. {len(hallucinated)} hallucinated."
        ),
    }


# ---------------------------------------------------------------------------
# Eval B: priority calibration (LLM judge)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """You are an experienced product manager grading the priority
calibration of a peer's PM report on an LLM observability product.

For each backlog row you receive, you have:
  - the assigned priority (P0–P3),
  - the theme + summary the PM wrote,
  - the metadata for every cited issue/discussion (title, body excerpt,
    reactions, comments, recency, labels, kind).

Priority rubric (same one the PM used):
  P0 = critical now: data loss, security, broken core flows, severe
       regression, or asks blocking multiple loud users.
  P1 = high this quarter: widely-requested features, important bugs with
       workarounds, top themes across discussions.
  P2 = medium next quarter: nice-to-haves with real demand, narrower bugs.
  P3 = low / backlog: single-asker requests, edge cases, polish.

Output strict JSON only, no prose:
{
  "verdict": "correct" | "too_high" | "too_low",
  "suggested_priority": "P0" | "P1" | "P2" | "P3",
  "score": 1-5,           // 5 = perfectly calibrated, 1 = wildly off
  "reasoning": "1-2 sentences citing concrete metadata"
}
"""


def parse_backlog(report: str) -> list[dict[str, Any]]:
    """Extract rows from the Prioritized Backlog table."""
    rows: list[dict[str, Any]] = []
    for m in BACKLOG_ROW_RE.finditer(report):
        priority, theme, summary, refs_cell = m.groups()
        # skip the header row if it sneaks through
        if theme.lower().startswith("theme") or "---" in theme:
            continue
        refs = sorted(set(int(c) for c in CITATION_RE.findall(refs_cell)))
        if not refs:
            continue
        rows.append({"priority": priority, "theme": theme, "summary": summary, "refs": refs})
    return rows


def judge_row(client, row: dict[str, Any], known: dict[int, dict[str, Any]]) -> dict[str, Any]:
    cited_meta = []
    for ref in row["refs"]:
        item = known.get(ref)
        if not item:
            continue
        cited_meta.append({
            "number": ref,
            "source": item["source"],
            "kind": item.get("kind"),
            "title": item.get("title"),
            "reactions": item.get("reactions"),
            "comments": item.get("comments"),
            "updated_at": item.get("updated_at"),
            "labels": item.get("labels"),
            "body_excerpt": (item.get("body_excerpt") or "")[:400],
        })

    if not cited_meta:
        return {
            "priority": row["priority"],
            "theme": row["theme"],
            "verdict": "skipped",
            "suggested_priority": row["priority"],
            "score": None,
            "reasoning": "no resolvable refs (likely hallucinated — counted in eval A)",
        }

    user_msg = (
        f"PM-assigned priority: {row['priority']}\n"
        f"Theme: {row['theme']}\n"
        f"Summary: {row['summary']}\n\n"
        f"Cited issues/discussions:\n```json\n{json.dumps(cited_meta, indent=2)}\n```"
    )

    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=400,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # strip fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        verdict = {"verdict": "parse_error", "raw": text, "score": None}

    return {
        "priority": row["priority"],
        "theme": row["theme"],
        "refs": row["refs"],
        **verdict,
    }


def eval_priority_calibration(report: str, raw: dict[str, Any], anth_key: str) -> dict[str, Any]:
    import anthropic
    client = anthropic.Anthropic(api_key=anth_key)
    known = build_known_numbers(raw)
    rows = parse_backlog(report)

    judged = [judge_row(client, r, known) for r in rows]

    scored = [j for j in judged if isinstance(j.get("score"), (int, float))]
    avg_score = round(sum(j["score"] for j in scored) / len(scored), 2) if scored else None
    correct = sum(1 for j in judged if j.get("verdict") == "correct")
    too_high = sum(1 for j in judged if j.get("verdict") == "too_high")
    too_low = sum(1 for j in judged if j.get("verdict") == "too_low")

    return {
        "name": "priority_calibration",
        "judge_model": JUDGE_MODEL,
        "rows_evaluated": len(judged),
        "avg_score_1_to_5": avg_score,
        "correct": correct,
        "too_high": too_high,
        "too_low": too_low,
        "per_row": judged,
    }


# ---------------------------------------------------------------------------
# Eval C: top-signal coverage
# ---------------------------------------------------------------------------

def eval_top_signal_coverage(report: str, raw: dict[str, Any], k: int = TOP_K) -> dict[str, Any]:
    """Of the top-K highest-scored input items, what fraction appear in the report?"""
    pool = [(i["score"], i["number"], i["kind"], i["title"]) for i in raw.get("issues", [])]
    pool += [(d["score"], d["number"], d["kind"], d["title"]) for d in raw.get("discussions", [])]
    pool.sort(key=lambda t: t[0], reverse=True)
    top = pool[:k]

    cited_nums = set(int(m.group(1)) for m in CITATION_RE.finditer(report))
    mentioned = [t for t in top if t[1] in cited_nums]
    missed = [t for t in top if t[1] not in cited_nums]

    score = round(len(mentioned) / k, 4) if k else 1.0
    return {
        "name": "top_signal_coverage",
        "score": score,
        "passed": score >= 0.8,
        "k": k,
        "mentioned_count": len(mentioned),
        "missed_count": len(missed),
        "missed_items": [
            {"number": n, "score": s, "kind": kind, "title": title[:80]}
            for s, n, kind, title in missed
        ],
        "explanation": (
            f"{len(mentioned)}/{k} of the top-scored input items are mentioned "
            f"in the report. {len(missed)} top items were skipped."
        ),
    }


# ---------------------------------------------------------------------------
# Eval D: structure compliance
# ---------------------------------------------------------------------------

def eval_structure_compliance(report: str) -> dict[str, Any]:
    sections = split_by_h2(report)
    present = []
    missing = []
    for required in REQUIRED_SECTIONS:
        if find_section(sections, required) is not None:
            present.append(required)
        else:
            missing.append(required)
    total = len(REQUIRED_SECTIONS)
    score = round(len(present) / total, 4) if total else 1.0
    return {
        "name": "structure_compliance",
        "score": score,
        "passed": score == 1.0,
        "required_count": total,
        "present": present,
        "missing": missing,
        "explanation": (
            f"{len(present)}/{total} required H2 sections present. "
            + (f"Missing: {', '.join(missing)}." if missing else "All present.")
        ),
    }


# ---------------------------------------------------------------------------
# Eval E: bug-vs-feature classification agreement
# ---------------------------------------------------------------------------

def _cited_in(section_body: str, known: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    nums = sorted(set(int(m.group(1)) for m in CITATION_RE.finditer(section_body)))
    return [{"number": n, **known[n]} for n in nums if n in known]


def eval_bug_vs_feature_classification(report: str, known: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Agreement between the report's grouping and the deterministic kind tag.

    Citations under "Top Pain Points" are expected to map to items the
    deterministic scorer labeled `bug`; "Top Feature Asks" should map to
    `feature`. Items labeled `other` are excluded from the denominator
    (the keyword-based classifier is too coarse to be authoritative there).
    """
    sections = split_by_h2(report)
    pain_body = find_section(sections, PAIN_POINTS_SECTION) or ""
    feat_body = find_section(sections, FEATURE_ASKS_SECTION) or ""

    pain_items = _cited_in(pain_body, known)
    feat_items = _cited_in(feat_body, known)

    pain_eval = [i for i in pain_items if i.get("kind") in ("bug", "feature")]
    feat_eval = [i for i in feat_items if i.get("kind") in ("bug", "feature")]

    pain_correct = [i for i in pain_eval if i["kind"] == "bug"]
    pain_wrong = [i for i in pain_eval if i["kind"] == "feature"]
    feat_correct = [i for i in feat_eval if i["kind"] == "feature"]
    feat_wrong = [i for i in feat_eval if i["kind"] == "bug"]

    total_eval = len(pain_eval) + len(feat_eval)
    total_correct = len(pain_correct) + len(feat_correct)
    score = round(total_correct / total_eval, 4) if total_eval else 1.0

    def _strip(items):
        return [{"number": i["number"], "kind": i["kind"], "title": (i.get("title") or "")[:80]}
                for i in items]

    return {
        "name": "bug_vs_feature_classification",
        "score": score,
        "passed": score >= 0.8,
        "total_evaluated": total_eval,
        "total_correct": total_correct,
        "pain_points_under_section": len(pain_items),
        "pain_points_classifier_label_bug": len(pain_correct),
        "pain_points_classifier_label_feature": len(pain_wrong),
        "feature_asks_under_section": len(feat_items),
        "feature_asks_classifier_label_feature": len(feat_correct),
        "feature_asks_classifier_label_bug": len(feat_wrong),
        "disagreements": {
            "pain_points_classified_as_feature": _strip(pain_wrong),
            "feature_asks_classified_as_bug": _strip(feat_wrong),
        },
        "explanation": (
            f"Of {total_eval} citations under Pain Points / Feature Asks with a "
            f"definitive bug/feature label from the deterministic classifier, "
            f"{total_correct} agree with the report's grouping."
        ),
    }


# ---------------------------------------------------------------------------
# Tracing helpers
# ---------------------------------------------------------------------------

def log_evaluator_span(tracer, name: str, result: dict[str, Any]) -> None:
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    with tracer.start_as_current_span(f"eval.{name}") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.EVALUATOR.value,
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, REPORT_PATH.name)
        score = result.get("score") or result.get("avg_score_1_to_5")
        if score is not None:
            span.set_attribute("eval.score", float(score))
        for k in ("total_citations", "grounded", "hallucinated_count",
                  "rows_evaluated", "correct", "too_high", "too_low",
                  "k", "mentioned_count", "missed_count",
                  "required_count", "total_evaluated", "total_correct"):
            if k in result and result[k] is not None:
                span.set_attribute(f"eval.{k}", result[k])
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(result)[:4000])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = Path("/Users/rishsharma/Claudecode/spark-app/.env.local")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ERROR: no ANTHROPIC_API_KEY.")


def print_scoreboard(a: dict[str, Any], b: dict[str, Any],
                     c: dict[str, Any], d: dict[str, Any], e: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(" Phoenix PM Agent — Eval Scoreboard")
    print("=" * 78)

    print(f"\n[A] citation_groundedness: {a['score']:.2%}  "
          f"({a['grounded']}/{a['total_citations']} grounded, "
          f"{a['hallucinated_count']} hallucinated)")
    if a["hallucinated_numbers"]:
        print(f"    hallucinated: {', '.join('#' + str(n) for n in a['hallucinated_numbers'])}")

    print(f"\n[B] priority_calibration: avg {b['avg_score_1_to_5']}/5  "
          f"({b['correct']} correct, {b['too_high']} too-high, {b['too_low']} too-low "
          f"out of {b['rows_evaluated']})")
    print(f"    judge: {b['judge_model']}")
    print("    Per-row verdicts:")
    for row in b["per_row"]:
        score = row.get("score")
        score_s = f"{score}/5" if score is not None else "skip"
        verdict = row.get("verdict", "?")
        suggested = row.get("suggested_priority", "")
        sug_s = f" -> {suggested}" if suggested and suggested != row["priority"] else ""
        print(f"      {row['priority']}{sug_s:<6}  {score_s:<6}  {verdict:<12}  {row['theme'][:50]}")

    print(f"\n[C] top_signal_coverage (top-{c['k']}): {c['score']:.2%}  "
          f"({c['mentioned_count']}/{c['k']} mentioned, {c['missed_count']} missed)")
    if c["missed_items"]:
        print("    missed top items:")
        for item in c["missed_items"][:10]:
            print(f"      #{item['number']:<6} ({item['kind']:<7}) score={item['score']:<6.2f}  {item['title']}")

    print(f"\n[D] structure_compliance: {d['score']:.2%}  "
          f"({len(d['present'])}/{d['required_count']} required H2 sections)")
    if d["missing"]:
        print(f"    missing: {', '.join(d['missing'])}")

    print(f"\n[E] bug_vs_feature_classification: {e['score']:.2%}  "
          f"({e['total_correct']}/{e['total_evaluated']} agree with deterministic classifier)")
    pain_d = e["disagreements"]["pain_points_classified_as_feature"]
    feat_d = e["disagreements"]["feature_asks_classified_as_bug"]
    if pain_d:
        print(f"    under Pain Points but classifier said `feature`:")
        for i in pain_d[:8]:
            print(f"      #{i['number']:<6}  {i['title']}")
    if feat_d:
        print(f"    under Feature Asks but classifier said `bug`:")
        for i in feat_d[:8]:
            print(f"      #{i['number']:<6}  {i['title']}")
    print()


def main() -> None:
    raw = load_raw()
    report = load_report()
    anth_key = get_anthropic_key()

    tracer_provider = setup_tracing(project_name="phoenix-pm-evals")

    from opentelemetry import trace
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("pm_report_evals.run") as root:
        root.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )
        root.set_attribute(SpanAttributes.INPUT_VALUE, REPORT_PATH.name)

        known = build_known_numbers(raw)
        print(f"loaded report ({len(report)} chars), raw signal "
              f"({len(known)} known issues+discussions)", file=sys.stderr)

        print("running eval A: citation_groundedness ...", file=sys.stderr)
        result_a = eval_citation_groundedness(report, known)
        log_evaluator_span(tracer, "citation_groundedness", result_a)

        print("running eval C: top_signal_coverage ...", file=sys.stderr)
        result_c = eval_top_signal_coverage(report, raw, k=TOP_K)
        log_evaluator_span(tracer, "top_signal_coverage", result_c)

        print("running eval D: structure_compliance ...", file=sys.stderr)
        result_d = eval_structure_compliance(report)
        log_evaluator_span(tracer, "structure_compliance", result_d)

        print("running eval E: bug_vs_feature_classification ...", file=sys.stderr)
        result_e = eval_bug_vs_feature_classification(report, known)
        log_evaluator_span(tracer, "bug_vs_feature_classification", result_e)

        print("running eval B: priority_calibration (LLM judge) ...", file=sys.stderr)
        result_b = eval_priority_calibration(report, raw, anth_key)
        log_evaluator_span(tracer, "priority_calibration", result_b)

        root.set_attribute("eval.citation_groundedness", float(result_a["score"]))
        root.set_attribute("eval.top_signal_coverage", float(result_c["score"]))
        root.set_attribute("eval.structure_compliance", float(result_d["score"]))
        root.set_attribute("eval.bug_vs_feature_classification", float(result_e["score"]))
        if result_b["avg_score_1_to_5"] is not None:
            root.set_attribute("eval.priority_calibration_avg", float(result_b["avg_score_1_to_5"]))

    shutdown_tracing(tracer_provider)
    print_scoreboard(result_a, result_b, result_c, result_d, result_e)

    # also dump full results for the curious
    out_path = REPO_ROOT / "evals" / "last_eval_results.json"
    out_path.write_text(json.dumps(
        {"A": result_a, "B": result_b, "C": result_c, "D": result_d, "E": result_e},
        indent=2,
    ))
    print(f"full results -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
