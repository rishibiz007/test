#!/usr/bin/env python3
"""PM agent for arize-ai/phoenix.

Pulls recent GitHub issues, discussions, and releases, scores each item for
priority (bug vs feature, reactions, comments, recency), then asks
Claude Opus 4.7 to synthesize a markdown PM report ordered P0–P3.

Auth:
  - GitHub: reads token from `gh auth token` (falls back to GITHUB_TOKEN env).
  - Anthropic: reads ANTHROPIC_API_KEY env (falls back to spark-app/.env.local).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from instrumentation import setup_tracing, shutdown_tracing

REPO_OWNER = "Arize-ai"
REPO_NAME = "phoenix"
GITHUB_API = "https://api.github.com"
GITHUB_GRAPHQL = "https://api.github.com/graphql"

ISSUE_LIMIT = 150       # most-recently-updated issues to inspect
DISCUSSION_LIMIT = 80   # most-recently-updated discussions
RELEASE_LIMIT = 15      # latest releases
RECENT_DAYS_WINDOW = 90

OPUS_MODEL = "claude-opus-4-7"

OUTPUT_PATH = Path(__file__).parent / "phoenix_pm_report.md"
RAW_DUMP_PATH = Path(__file__).parent / "phoenix_raw.json"


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def get_github_token() -> str:
    env_tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if env_tok:
        return env_tok
    try:
        out = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    sys.exit("ERROR: no GitHub token. Set GITHUB_TOKEN or run `gh auth login`.")


def get_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = Path("/Users/rishsharma/Claudecode/spark-app/.env.local")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ERROR: no ANTHROPIC_API_KEY. Set the env var.")


# ---------------------------------------------------------------------------
# GitHub fetchers
# ---------------------------------------------------------------------------

def gh_rest(token: str, path: str, params: dict[str, Any] | None = None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    url = f"{GITHUB_API}{path}"
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
        wait = max(1, reset - int(time.time()))
        print(f"  rate-limited — sleeping {wait}s", file=sys.stderr)
        time.sleep(min(wait, 60))
        resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def gh_graphql(token: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(
        GITHUB_GRAPHQL,
        headers=headers,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


def fetch_issues(token: str, limit: int) -> list[dict[str, Any]]:
    """Pull open issues sorted by most recently updated, excluding PRs."""
    issues: list[dict[str, Any]] = []
    per_page = 50
    pages = math.ceil(limit / per_page)
    for page in range(1, pages + 1):
        batch = gh_rest(
            token,
            f"/repos/{REPO_OWNER}/{REPO_NAME}/issues",
            {
                "state": "open",
                "sort": "updated",
                "direction": "desc",
                "per_page": per_page,
                "page": page,
            },
        )
        for item in batch:
            if "pull_request" in item:
                continue
            issues.append(item)
            if len(issues) >= limit:
                return issues
        if len(batch) < per_page:
            break
    return issues


def fetch_releases(token: str, limit: int) -> list[dict[str, Any]]:
    return gh_rest(
        token,
        f"/repos/{REPO_OWNER}/{REPO_NAME}/releases",
        {"per_page": limit},
    )


DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: $first, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        url
        body
        createdAt
        updatedAt
        upvoteCount
        answerChosenAt
        category { name }
        author { login }
        comments { totalCount }
        reactions { totalCount }
        labels(first: 10) { nodes { name } }
      }
    }
  }
}
"""


def fetch_discussions(token: str, limit: int) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    cursor: str | None = None
    while len(nodes) < limit:
        page_size = min(50, limit - len(nodes))
        data = gh_graphql(
            token,
            DISCUSSIONS_QUERY,
            {"owner": REPO_OWNER, "name": REPO_NAME, "first": page_size, "after": cursor},
        )
        disc = data["repository"]["discussions"]
        nodes.extend(disc["nodes"])
        if not disc["pageInfo"]["hasNextPage"]:
            break
        cursor = disc["pageInfo"]["endCursor"]
    return nodes[:limit]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

BUG_KEYWORDS = ("bug", "broken", "error", "crash", "regression", "doesn't work", "fails", "exception")
FEATURE_KEYWORDS = ("feature", "request", "enhancement", "proposal", "support for", "would be nice", "ability to", "allow")


def classify_kind(title: str, body: str, labels: list[str]) -> str:
    text = f"{title} {body[:500]}".lower()
    label_text = " ".join(labels).lower()
    if any(k in label_text for k in ("bug", "regression", "defect")):
        return "bug"
    if any(k in label_text for k in ("feature", "enhancement", "request")):
        return "feature"
    if any(k in text for k in BUG_KEYWORDS):
        return "bug"
    if any(k in text for k in FEATURE_KEYWORDS):
        return "feature"
    return "other"


def recency_weight(updated_at: str) -> float:
    """1.0 if updated today, decays to ~0.3 at RECENT_DAYS_WINDOW."""
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.5
    age_days = (datetime.now(timezone.utc) - updated).days
    return max(0.2, math.exp(-age_days / RECENT_DAYS_WINDOW))


@dataclass
class ScoredItem:
    kind: str              # bug | feature | other
    source: str            # issue | discussion
    number: int
    title: str
    url: str
    body_excerpt: str
    labels: list[str]
    reactions: int
    comments: int
    updated_at: str
    created_at: str
    author: str
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)


def score_issue(issue: dict[str, Any]) -> ScoredItem:
    labels = [l["name"] for l in issue.get("labels", [])]
    body = issue.get("body") or ""
    kind = classify_kind(issue["title"], body, labels)
    reactions = issue.get("reactions", {}).get("total_count", 0)
    comments = issue.get("comments", 0)
    recency = recency_weight(issue["updated_at"])

    kind_weight = {"bug": 1.4, "feature": 1.0, "other": 0.8}[kind]
    raw = (reactions * 3 + comments * 2 + 1) * kind_weight * recency
    return ScoredItem(
        kind=kind,
        source="issue",
        number=issue["number"],
        title=issue["title"],
        url=issue["html_url"],
        body_excerpt=body[:600],
        labels=labels,
        reactions=reactions,
        comments=comments,
        updated_at=issue["updated_at"],
        created_at=issue["created_at"],
        author=(issue.get("user") or {}).get("login", "unknown"),
        score=round(raw, 2),
        score_breakdown={
            "reactions": reactions,
            "comments": comments,
            "kind_weight": kind_weight,
            "recency": round(recency, 3),
        },
    )


def score_discussion(disc: dict[str, Any]) -> ScoredItem:
    labels = [n["name"] for n in (disc.get("labels", {}) or {}).get("nodes", [])]
    body = disc.get("body") or ""
    kind = classify_kind(disc["title"], body, labels + [disc["category"]["name"]])
    reactions = (disc.get("reactions") or {}).get("totalCount", 0) + (disc.get("upvoteCount") or 0)
    comments = (disc.get("comments") or {}).get("totalCount", 0)
    recency = recency_weight(disc["updatedAt"])

    kind_weight = {"bug": 1.3, "feature": 1.1, "other": 0.9}[kind]
    raw = (reactions * 3 + comments * 2 + 1) * kind_weight * recency
    return ScoredItem(
        kind=kind,
        source="discussion",
        number=disc["number"],
        title=disc["title"],
        url=disc["url"],
        body_excerpt=body[:600],
        labels=labels + [f"category:{disc['category']['name']}"],
        reactions=reactions,
        comments=comments,
        updated_at=disc["updatedAt"],
        created_at=disc["createdAt"],
        author=((disc.get("author") or {}) or {}).get("login", "unknown") or "unknown",
        score=round(raw, 2),
        score_breakdown={
            "reactions+upvotes": reactions,
            "comments": comments,
            "kind_weight": kind_weight,
            "recency": round(recency, 3),
        },
    )


# ---------------------------------------------------------------------------
# Claude synthesis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior product manager writing a prioritization
report for the Arize Phoenix open-source LLM observability product.

Your job: read the structured GitHub signal below (scored issues, scored
discussions, recent releases) and produce a sharp, actionable markdown PM
report.

Priority rubric:
- P0 (Critical, this sprint): data loss, security, broken core flows, severe
  regressions, or asks blocking multiple loud users.
- P1 (High, this quarter): widely-requested features, important bugs with
  workarounds, top themes across discussions.
- P2 (Medium, next quarter): nice-to-haves with real demand, narrower bugs.
- P3 (Low / backlog): single-asker requests, edge cases, polish.

Use the provided score as one signal, but weight your own judgement on
severity, blast radius, and strategic fit for an LLM-observability product.
Group similar issues/discussions into THEMES rather than listing every item.
Always cite the underlying issue/discussion numbers (e.g. #1234) so the PM
can drill in.

Output ONLY the markdown report. No preamble.
"""


REPORT_TEMPLATE = """Write the report with this structure:

# Phoenix PM Report — {today}

## Executive Summary
3–5 bullets. What's on fire, what users keep asking for, what shipped recently.

## Recent Release Context
Brief — what landed in the last few releases. Helps frame whether top asks are
already addressed.

## Top Pain Points (Bugs & Friction)
Grouped themes, each with: title, P0–P3 tag, 2–4 sentence summary,
cited issue/discussion numbers, suggested next step.

## Top Feature Asks
Same structure as pain points but for enhancement-style requests.

## Cross-Cutting Themes
Patterns that span bugs + features (e.g. "tracing UX", "auth/RBAC", "OTel
compatibility"). 3–6 themes max. Each gets a P-tag and short rationale.

## Prioritized Backlog (P0 → P3)
A single ordered table: Priority | Theme | Summary | Top references.

## Notable Single-Asker Asks
Quick list — small but interesting signal worth tracking.
"""


def synthesize_report(api_key: str, payload: dict[str, Any]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_message = (
        REPORT_TEMPLATE.format(today=datetime.now().strftime("%Y-%m-%d"))
        + "\n\nHere is the scored signal as JSON:\n\n```json\n"
        + json.dumps(payload, indent=2)[:180_000]
        + "\n```"
    )

    resp = client.messages.create(
        model=OPUS_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[1/5] auth ...", file=sys.stderr)
    gh_token = get_github_token()
    anth_key = get_anthropic_key()

    tracer_provider = setup_tracing(project_name="phoenix-pm-agent")

    from opentelemetry import trace
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("phoenix_pm_agent.run") as root_span:
        root_span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )
        root_span.set_attribute(SpanAttributes.INPUT_VALUE, f"{REPO_OWNER}/{REPO_NAME}")
        _run_agent(gh_token, anth_key, root_span)
    shutdown_tracing(tracer_provider)


def _run_agent(gh_token: str, anth_key: str, root_span) -> None:
    from openinference.semconv.trace import SpanAttributes

    print(f"[2/5] fetching up to {ISSUE_LIMIT} issues ...", file=sys.stderr)
    issues = fetch_issues(gh_token, ISSUE_LIMIT)
    print(f"      got {len(issues)} issues", file=sys.stderr)

    print(f"[3/5] fetching up to {DISCUSSION_LIMIT} discussions ...", file=sys.stderr)
    discussions = fetch_discussions(gh_token, DISCUSSION_LIMIT)
    print(f"      got {len(discussions)} discussions", file=sys.stderr)

    print(f"[4/5] fetching last {RELEASE_LIMIT} releases ...", file=sys.stderr)
    releases = fetch_releases(gh_token, RELEASE_LIMIT)
    print(f"      got {len(releases)} releases", file=sys.stderr)

    scored_issues = sorted(
        [score_issue(i) for i in issues], key=lambda s: s.score, reverse=True
    )
    scored_discussions = sorted(
        [score_discussion(d) for d in discussions], key=lambda s: s.score, reverse=True
    )

    release_summaries = [
        {
            "tag": r.get("tag_name"),
            "name": r.get("name"),
            "published_at": r.get("published_at"),
            "body_excerpt": (r.get("body") or "")[:1200],
            "url": r.get("html_url"),
        }
        for r in releases
    ]

    payload = {
        "repo": f"{REPO_OWNER}/{REPO_NAME}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "issues": len(scored_issues),
            "discussions": len(scored_discussions),
            "releases": len(release_summaries),
        },
        "releases": release_summaries,
        "issues": [asdict(s) for s in scored_issues],
        "discussions": [asdict(s) for s in scored_discussions],
    }

    RAW_DUMP_PATH.write_text(json.dumps(payload, indent=2))
    print(f"      raw signal -> {RAW_DUMP_PATH}", file=sys.stderr)

    print(f"[5/5] asking {OPUS_MODEL} to write the report ...", file=sys.stderr)
    report = synthesize_report(anth_key, payload)
    OUTPUT_PATH.write_text(report)
    print(f"\nReport written to: {OUTPUT_PATH}", file=sys.stderr)

    root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, report[:4000])
    root_span.set_attribute("phoenix_pm.issues_count", len(scored_issues))
    root_span.set_attribute("phoenix_pm.discussions_count", len(scored_discussions))
    root_span.set_attribute("phoenix_pm.releases_count", len(release_summaries))


if __name__ == "__main__":
    main()
