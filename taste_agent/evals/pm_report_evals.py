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
                  "rows_evaluated", "correct", "too_high", "too_low"):
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


def print_scoreboard(a: dict[str, Any], b: dict[str, Any]) -> None:
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
    print()
    print("    Per-row verdicts:")
    for row in b["per_row"]:
        score = row.get("score")
        score_s = f"{score}/5" if score is not None else "skip"
        verdict = row.get("verdict", "?")
        suggested = row.get("suggested_priority", "")
        sug_s = f" -> {suggested}" if suggested and suggested != row["priority"] else ""
        print(f"      {row['priority']}{sug_s:<6}  {score_s:<6}  {verdict:<12}  {row['theme'][:50]}")
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

        print("running eval B: priority_calibration (LLM judge) ...", file=sys.stderr)
        result_b = eval_priority_calibration(report, raw, anth_key)
        log_evaluator_span(tracer, "priority_calibration", result_b)

        root.set_attribute("eval.citation_groundedness", float(result_a["score"]))
        if result_b["avg_score_1_to_5"] is not None:
            root.set_attribute("eval.priority_calibration_avg", float(result_b["avg_score_1_to_5"]))

    shutdown_tracing(tracer_provider)
    print_scoreboard(result_a, result_b)

    # also dump full results for the curious
    out_path = REPO_ROOT / "evals" / "last_eval_results.json"
    out_path.write_text(json.dumps({"A": result_a, "B": result_b}, indent=2))
    print(f"full results -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
