# Phoenix PM Report — 2026-06-03

## Executive Summary
- **PXI (Phoenix's in-app agent) is dominating the roadmap.** The vast majority of recent issues and releases are PXI tools, skills, and UX polish (#13572, #13570, #13616). This is a strategic bet, but it's crowding out core observability bug fixes.
- **Cost/token accounting bugs are eroding trust.** Two independent reports of inflated/incorrect cost metrics (#13379 cached tokens charged as output; #12768 dashboard triple-counts tokens). These are P0 — users cannot rely on the numbers Phoenix shows them.
- **PXI agent reliability issues are piling up fast.** Span ID hallucinations (#13512), opaque GraphQL errors that feed the hallucinations (#13513), empty tool-call inputs (#13514), timeouts on reasoning models (#13569). These are self-inflicted by the rapid PXI build-out and need a hardening sprint.
- **Auth/RBAC gap blocking enterprise adoption.** OAuth2 access-token claim mapping (#13557) was closed `wontfix` despite real demand; combined with long-standing per-project RBAC ask (#9565), this is a recurring enterprise blocker.
- **Recent shipping velocity is exceptional.** 15 releases in 2 weeks (v15.10 → v17.2), with major v16.0 Code Evaluators launch and v17.0 admin system settings. Migration discipline is good (MIGRATION.md called out on breaking changes).

## Recent Release Context
- **v17.0–v17.2 (Jun 2–3):** New admin-managed system settings (assistant enablement, trace recording policy) — a breaking change with migration guide. PXI agent gets route info tool, dataset load tool, LLM-evaluator authoring, skills display.
- **v16.0–v16.6 (May 21 – Jun 2):** Headline launch — **Sandboxed Code Evaluators** (Python/TS evaluate() functions, server-side execution, composite scoring, juries). Plus dashboards route, PXI floating panel, model switching, web search toggle, Claude Opus 4.8 support.
- **v15.10–v15.12 (May 15–21):** Generative UI in agent chats, OTel GenAI semconv conversion (#13250-ish), token-count confinement to LLM spans at ingestion (#13433 — relevant to #12768), built-in model token-price updates.
- **Net:** Code Evaluators and PXI agent are the strategic thrust. Core trace/cost plumbing got partial fixes but not all reported regressions are addressed.

## Top Pain Points (Bugs & Friction)

### 1. Cost & token accounting is wrong in multiple places — **P0**
Phoenix is showing users inflated/incorrect spend, the #1 trust-breaker for an observability product. Cached tokens are being billed as output tokens in the UI (#13379, v15.11.1), and the dashboard overview triple-counts tokens by summing nested spans (#12768, v14.6.0). The v16.x ingestion fix (#13433) addressed token-count confinement to LLM spans but doesn't fully cover the dashboard rollup or cache pricing.
**Next step:** Audit all cost/token rollup paths end-to-end; add regression tests with multi-hop nested spans and Anthropic-style cache-read tokens; publish a "what changed in v16.x token accounting" note.

### 2. PXI agent reliability cluster — **P0**
The PXI agent ships fast but fails noisily: hallucinated span/trace IDs in batch annotate (#13512), empty `{}` tool inputs (#13514), opaque "an unexpected error occurred" GraphQL responses that perpetuate the hallucinations (#13513), timeouts on `gpt-5.5` reasoning models (#13569), fabricated UI links (#13528, #13609), and the annotate panel failing to open in long tool chains (#13580). Docs MCP failures also stall every assistant turn even after the startup hotfix (#13599).
**Next step:** One-sprint "PXI hardening" theme — tighten tool-input validation, return actionable GraphQL errors with resolvable IDs, raise reasoning-model timeouts, and add an integration test suite for multi-tool-call sequences.

### 3. Self-hosting startup & platform friction — **P1**
Phoenix can be hard to stand up: Apple Silicon + podman SIGILL on `cryptography 47.0.0` (#12941), startup blocked on WASM binary download with no timeout (#13382), sandbox per-execute timeout not enforced across 4/6 backends (#13313). These hit new evaluators users hard right after the v16.0 Sandboxed Code Evaluators launch.
**Next step:** Pin/replace cryptography on ARM images; make WASM prefetch lazy or gated by feature flag with a hard timeout; complete sandbox timeout enforcement.

### 4. Playground span replay & provider gaps — **P1**
Anthropic memory-tool span replay fails because `tool_use` isn't paired with synthesized `tool_result` (#12975). Gemini playground calls drop tool calls from output messages (#12971, inconsistent with Nova). PXI `write_prompt_tools` approval diffs go stale when the provider changes mid-pending (#13626).
**Next step:** Treat playground replay as a tier-1 supported flow; add per-provider replay golden tests; expire pending diffs on provider change.

### 5. Trace/cost data integrity edge cases — **P2**
Docs gap audits (#13212, #13457) and a user report (#13521) flag that Phoenix only reads `session.id`/`user.id` from root spans, breaking integrations where Phoenix is a second exporter in an existing OTel pipeline. Slow filtering on `spans.attributes` at scale (~100k/day, discussion #11233) needs indexing guidance.
**Next step:** Document/relax root-span-only assumption for sessions; publish a perf/indexing playbook for high-volume self-hosted users.

### 6. Prompt-save UX & error messaging — **P2 / polish**
Save-a-prompt error is hard to read, off-screen, and unfriendly (#9940). Hostname vs. collector endpoint confuses users (#9971). Bug in `ai_evals_hw4_solution.ipynb` tutorial yields wrong MRR/recall (#13394).
**Next step:** Friction sweep on prompt-save flow and onboarding terminology; refresh tutorial notebooks against current package versions.

## Top Feature Asks

### 1. OAuth2 / RBAC expansion — **P1**
Two converging asks: map RBAC roles from OAuth2 **access tokens** (not just ID tokens/userinfo) per security best practice (#13557 — currently marked `wontfix` but has real demand), and **per-project RBAC** to segment dev/QA/prod users (discussion #9565, 4 reactions, long-standing). This is recurring enterprise table stakes.
**Next step:** Reopen #13557 conversation — even minimal access-token JMESPath support is high-leverage. Scope a per-project RBAC design doc this quarter.

### 2. PXI agent capability expansion (the big epic) — **P1**
The PXI Playground Tools & Skills epic (#13572) and Tracing Skills epic (#13570) span ~20+ sub-issues: dataset CRUD (#13616, #13600, #13605), prompt tool authoring & tool_choice (#13551, #13550), invocation param tuning (#13602), run lifecycle (#13606, #13607), experiment context (#13586, #13420), annotation config skills (#13575, #13546, #13515), generative charting (#13272). Strategically central but already starting to outpace quality (see PXI reliability cluster above).
**Next step:** Stage the epic into "P0 quality-bar" subset (must work reliably) vs. "P2 nice-to-have". Pair each capability with an eval (#13295 is the right pattern).

### 3. Vendor-neutral GenAI semantic conventions — **P1**
RFC discussion #13041 from a community contributor proposing a shared OTel GenAI semconv across Phoenix/Langfuse/OpenLLMetry. Phoenix already shipped GenAI semconv conversion in v15.10 — strong opportunity to take a leadership stance and avoid divergence.
**Next step:** Have a Phoenix engineer engage substantively on #13041; cross-link to v15.10 GenAI semconv work; publish Phoenix's stance.

### 4. Multi-hop RAG / agent eval surface — **P1**
Span-level eval to surface context-quality drops in multi-hop retrieval (#12552, top-scored issue, 8 comments). Also: GPA (Goal-Plan-Action) agent evaluators (#13111), entire chat-conversation eval (discussion #5654), multi-turn eval workflow (discussion #13393). Pattern: users want richer eval semantics for agents beyond per-span LLM-as-judge.
**Next step:** Bundle into an "agent evals" workstream that intersects with Code Evaluators (v16.0) — composite scoring is the right primitive; need recipes and built-ins.

### 5. External tracing/integration & DB support — **P2**
MariaDB support (#13482), Claude Code native OTel export integration (discussion #11153), encrypting span attributes client-side (discussion #9333), exporting token costs per project via API (discussion #11006), thread-based tracing for LangGraph breakpoints (discussion #5542).
**Next step:** Triage these into the integrations backlog; the "encrypt attributes" ask is worth a short design doc since data-residency requests will recur.

### 6. UX / polish for PXI and core app — **P2/P3**
Slash command for skills (#13304), input-focus on PXI open (#13627), custom time-range presets (#13192), filter eval-metric charts (#12781), trace annotations in trace view (#13296), beautiful empty states (#13597), divergent/continuous chart color (#13453). Mostly single-asker but a cumulative polish pass.

## Cross-Cutting Themes

| Theme | Priority | Rationale |
|---|---|---|
| **Cost/token correctness** | **P0** | Multiple reports across versions (#13379, #12768, #13212 audit gap), partial fixes in flight (#13433). Foundational to "observability" promise. |
| **PXI agent reliability** | **P0** | 10+ open bugs (#13512, #13513, #13514, #13569, #13580, #13599, #13609, #13528) all opened by Phoenix internals. Velocity is outpacing quality. |
| **PXI agent capability buildout** | **P1** | Epics #13572 and #13570 are the strategic bet; needs prioritization within the epic. |
| **Auth/RBAC for enterprise** | **P1** | #13557 (access tokens) + #9565 (project RBAC) recur with no clean answer. |
| **OTel/semconv compatibility** | **P1** | Cross-tool standardization (#13041) + multi-exporter setups (#13521) + GenAI semconv work in v15.10 all point at being the reference implementation. |
| **Self-host startup robustness** | **P2** | Apple Silicon (#12941), WASM prefetch (#13382), sandbox timeouts (#13313) — quick wins to remove first-run friction. |

## Prioritized Backlog (P0 → P3)

| Priority | Theme | Summary | Top References |
|---|---|---|---|
| **P0** | Cost/token correctness | Cached tokens billed as output; dashboard triple-counts tokens; audit all rollup paths | #13379, #12768, #13433, #13212 |
| **P0** | PXI agent reliability | ID hallucinations, opaque GraphQL errors, empty tool inputs, reasoning-model timeouts, docs MCP stalls | #13512, #13513, #13514, #13569, #13599, #13580 |
| **P0** | Sandbox eval safety | Per-execute timeout not enforced in 4/6 backends — security/DoS risk post-v16.0 | #13313, #13365 |
| **P1** | Self-host startup | Apple Silicon SIGILL, WASM prefetch blocks startup | #12941, #13382 |
| **P1** | OAuth2 access-token RBAC + per-project RBAC | Reopen access-token claim mapping; scope per-project RBAC | #13557, disc #9565 |
| **P1** | Playground replay & provider parity | Anthropic memory-tool, Gemini tool calls, provider-change diff staleness | #12975, #12971, #13626 |
| **P1** | PXI capability epics (staged) | Playground Tools & Skills + Tracing Skills epics — gate on quality | #13572, #13570, #13600, #13605, #13551, #13546 |
| **P1** | Multi-hop / agent evals | Span-level context-quality eval; GPA evaluators; multi-turn eval | #12552, #13111, disc #13393 |
| **P1** | OTel GenAI semconv leadership | Engage on vendor-neutral semconv RFC; document Phoenix stance | disc #13041, #13521 |
| **P2** | Cost API & per-project breakdown | Surface token/cost per project via REST | disc #11006, #11472 |
| **P2** | External tools client-side tracing | AI SDK / browser-executed tools lost backend TOOL spans | #13173 |
| **P2** | Integration breadth | MariaDB, Claude Code, envoy gateway, encryption | #13482, disc #11153, disc #9333, disc #10847 |
| **P2** | Onboarding/UX friction | Save-prompt error UI, hostname/collector naming, tutorial bug | #9940, #9971, #13394 |
| **P2** | PXI UX polish (themed) | Slash commands, focus on open, slot-aware modal behavior | #13304, #13627, #13598, #13428, #13594 |
| **P3** | Single-asker polish | Time-range presets, eval metric filter, branch session, custom icons | #13192, #12781, #13423, #13596 |
| **P3** | Generative charting/UI skill | Phoenix-native dashboards generated by PXI | #13272, #13453 |

## Notable Single-Asker Asks
- **#10161** — Delinquent account messaging for Phoenix Cloud (interesting for billing/UX).
- **#13165** — Send PXI user_feedback annotations to remote Phoenix Cloud (dogfooding signal).
- **#13380** — Deprecate `baseURL` in favor of provider-specific endpoints (cleans up Azure/custom provider story).
- **#13116** — Ingestion-time middleware for AI Vercel conventions / GenAI / per-resource project routing.
- **disc #5542** — LangGraph breakpoints + thread-id continuation; touches a popular framework.
- **disc #6606** — Time-to-first-token latency for streaming LLM spans without dual spans (3 reactions, recurring).
- **disc #8625** — Old logs dropped under high throughput; needs a high-volume ingestion guide.
- **disc #12818** — Assay external-consumer integration on span annotations; signal that the annotation API is being adopted as a public seam.
- **#11472** — RAG failure-mode checklist docs page (contributor offered to draft a PR — easy yes).
- **#13525 / #13537** — HVTracker integration requests; ignorable but worth a templated response.