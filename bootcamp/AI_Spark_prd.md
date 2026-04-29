# ICE Breaker — PRD v1

> A networking copilot that gives job seekers something real to say in their next coffee chat.

**Team:** Adi & Rishi

---

## Table of Contents

- [Mission](#mission)
- [Why Invest?](#why-invest)
- [User Ecosystem](#user-ecosystem)
- [Segments](#segments)
- [Focus Persona](#focus-persona)
- [Pain Points](#pain-points)
- [Product Mission](#product-mission)
- [Solutions](#solutions)
- [MVP](#mvp)
- [Risks / Mitigation](#risks--mitigation)
- [Metrics](#metrics)
- [Long-Term Strategy](#long-term-strategy)

---

## Mission

Make every first conversation feel like the second one — by turning scattered public info about a person into 3-5 personal, timely, non-creepy talking points in under 30 seconds.

### North Star Metric (NSM)

**Sparked Conversations per WAU** — a topic was 👍'd, copied, or self-reported as "actually used."

**Supporting Metrics:**
- Time-to-Spark (< 30s)
- Topic hit rate (👍 / total rated)
- Week-4 retention
- Lookups per active user per week

---

## Why Invest?

### Macro

- Hybrid work killed casual context
- ~70% of jobs filled via referrals
- Loneliness / small-talk anxiety at all-time highs
- Networking volume rebounded past 2019

### Tech

- LLMs finally good at multi-source synthesis
- Agent UX normalized
- Inference cheap (~$0.02/lookup)
- Scraping/data tooling mature

**Team strengths:** Small, fast, dogfoodable during job search; narrow scope (one input → one output); showcases AI taste, not just plumbing; no two-sided network needed.

### Competition

**Tier 1 — Beat 2×:**
| Competitor | Weakness |
|---|---|
| Crystal Knows | Mechanical, sales-only |
| LinkedIn Premium AI | Generic, trapped inside LinkedIn |
| ChatGPT freeform | No integration, no memory |

**Tier 2 — Beat 10×:**
| Competitor | Notes |
|---|---|
| Microsoft Copilot + LinkedIn | Real long-term threat — 12-24 month window |
| Granola / Read.ai | Expanding into pre-meeting prep |
| New AI-native startup | Going viral in one segment first |

### Strategy

**Barriers:**
- Quality compounds with feedback data
- Trust / non-creepy tuning is hard
- Multi-source aggregation is grunt work that compounds

**Adoption Constraints:**
- Install friction (web app, no extension v1)
- LinkedIn ToS hostility (scraping at demo scale only)
- Creepiness line (public-data-only, sources cited, no protected attributes)

**Business Model Levers (later):**
- Freemium ($8–15/mo)
- Recruiter/sales SKU ($30–50)
- Event/conference partnerships
- B2B API

**Unit Economics:**
- ~$0.02–0.05 variable cost/lookup
- ~85% gross margin on paid
- Near-zero CAC if Chrome Web Store / Product Hunt / X loop hits

---

## User Ecosystem

**Demand:** Job seekers, founders, salespeople, recruiters, conference attendees, socially anxious professionals, students/early-career networkers.

**Supply:**
- LinkedIn (roles, posts, education)
- Public web (podcasts, blogs, talks, newsletters, GitHub)
- Twitter/X (current thinking)
- Recent activity (last 7d — the timeliness layer)
- Mutual context (later)

---

## Segments

### Demographic
- Early-career grads (22–28)
- Mid-career switchers (28–40)
- Returning-to-work professionals

### Behavioral
- Power (6+ chats/wk)
- Casual (1–2/wk)
- New (just starting)

### Motivational
- **Efficiency seekers** — save prep time
- **Value seekers** — better outcomes per chat
- **Community seekers** — build real network
- **Confidence seekers** — reduce anxiety

### Prioritization

Ranked by: market size × frequency × severity × feasibility

| Priority | Segment | Reasoning |
|---|---|---|
| **WIN** | Early/mid-career power users (efficiency + confidence motivated) | High freq, high severity, dogfoodable, reachable |
| **LATER** | Recruiters, founders | Lower urgency for v1 |

---

## Focus Persona

### "Networking Nikhil"

- **Age:** 26, MS CS grad, junior PM
- **Location:** Bay Area
- **Situation:** On H-1B/OPT clock; 4–8 coffee chats/wk hunting referrals
- **Behavior:** Spends 15–30 min prepping each chat and still walks in with generic questions; reads Lenny's, listens to Acquired; active on LinkedIn 3–5×/day

> *"I have a chat with a Senior PM at Stripe in 20 min and I've been staring at her LinkedIn for 10. I have nothing to actually say."*

**Later expansion:** Mid-career switchers → recruiters/sales (paid).

---

## Pain Points

Scored by **Severity × Frequency (1–5 scale)**:

| # | Pain Point | Score |
|---|---|---|
| P1 | Walking in with nothing personal to say | 5×5 = **25** |
| P2 | 20–30 min of prep yields generic résumé summary | 4×5 = **20** |
| P3 | Asking the same lazy questions every time | 4×4 = **16** |
| P4 | Cold outreach all sounds the same, gets ignored | 4×4 = **16** |
| P5 | Missing the timely thing (post/news from this week) | 5×3 = **15** |
| P6 | Info scattered across 6 sources | 3×5 = **15** |
| P7 | Forgetting prior conversations with same person | 5×3 = **15** |
| P8 | Awkward silence at chat minute 12 of 30 | 4×3 = **12** |
| P9 | Crossing the creepy line by mistake | 5×2 = **10** |

**v1 targets:** P1, P2, P5, P6.

---

## Product Mission

Be the best place for a job seeker to prep for a coffee chat in under a minute, and the most trusted layer between public info and human conversation.

---

## Solutions

Scored by **Effort × Impact**:

### High Impact / Low Effort — v1

| # | Solution |
|---|---|
| S1 | Paste LinkedIn URL → 3–5 categorized topics with sources |
| S2 | 👍/👎 with required quick-tags + optional free text on 👎 |
| S3 | Copy-to-clipboard per topic (implicit signal) |
| S4 | Lookup history (cached re-opens) |

### High Impact / High Effort — Later

| # | Solution |
|---|---|
| S5 | Chrome extension wrapper |
| S6 | Cold outreach message generator |
| S7 | Post-chat note + retention loop |

### Low Impact — Skip

- Mobile polish
- Auth depth
- Marketing site

---

## MVP

**Markets:** US, English-only.

**Segments:** Job seekers in tech (early-career power users); team dogfooding.

**Surfaces:** Web app (Next.js + Supabase + Vercel). Single input, single result page, history, admin dashboard. Desktop-first; mobile not broken.

**Partners:** None. Scraping at demo scale only. Anthropic/OpenAI for inference.

### Build Scope (2-week sprint)

```
URL input
  → server-side scrape (LinkedIn + web search + recent posts)
  → LLM returns structured topics w/ sources
  → render with 👍/👎, qualitative feedback on 👎, copy button
  → save lookup to history
  → admin dashboard for feedback review
```

---

## Risks / Mitigation

| Risk | Mitigation |
|---|---|
| **Accuracy** | Source-cite every topic; show profile snippet; cache for review |
| **Safety** | Hard-coded filter for protected attrs (health, religion, politics, family, ethnicity, age, orientation) at prompt + post-gen |
| **Creepiness** | Public-data-only; no inferences about non-public traits; clear footer messaging |
| **Latency** | < 30s end-to-end; parallelize scrapes; 24h cache; loading states that show progress |
| **Legal** | LinkedIn ToS — demo scale only; no production claims; document scope clearly in write-up |
| **Cost** | ~$0.05/lookup ceiling; rate-limit per user; small-model first, escalate only if quality fails |

---

## Metrics

### Engagement
- Activation: 1st lookup → 1st rating
- Retention: W1, W4 return rate
- Task completion: % of lookups with ≥1 👍 or copy
- Lookups per WAU

### Quality
- Topic hit rate (👍 / total rated)
- 👎 reason distribution
- Qualitative feedback volume + readability

### Guardrails
- Time-to-Spark P50 / P90
- Scrape success rate
- Sensitive-attribute filter false-negative rate (manual audit)
- $ / lookup
- Error rate

---

## Long-Term Strategy

If v1 works, future adjacencies (in rough order):

1. **Cold outreach generator** — same data, different output
2. **Post-chat memory + relationship CRM** — "the second meeting" feature
3. **Chrome extension on LinkedIn** — the magical surface
4. **Calendar-native auto-prep** — before every meeting automatically
5. **Recruiter/sales SKU** — paid tier, higher willingness to pay
6. **B2B API** — for CRMs / ATSes
7. **Ambient/wearable** — "whisper before the handshake"

> The wedge is job seekers. The endgame is the human-context layer for every professional conversation.
