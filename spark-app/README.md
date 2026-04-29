# Spark — Networking copilot

A Next.js web app that turns scattered public LinkedIn information into 3-5 personal, timely, non-creepy talking points for a coffee chat.

Built from the [ICE Breaker PRD v1](../bootcamp/AI_Spark_prd.md) and the hi-fi prototype in `bootcamp/Spark/`.

## What it does

1. **Onboarding** — paste your LinkedIn URL; Spark prefills your profile.
2. **Lookup** — paste a target's LinkedIn URL → triggers an Apify Actor that scrapes their profile + recent posts → Claude synthesizes 3-5 categorized topics with sources.
3. **Feedback** — 👍 / 👎 each topic. Down-votes require a reason and feed the eval set.
4. **History, Profile, Admin** — review past lookups, edit your profile, triage feedback.

## Stack

- Next.js 14 (App Router) + TypeScript
- React 18 client components, localStorage state
- `apify-client` to trigger LinkedIn scraping Actors
- `@anthropic-ai/sdk` (Claude Sonnet 4.6) for topic synthesis
- Plain CSS ported from the hi-fi prototype

## Setup

```bash
npm install
cp .env.example .env.local
# fill in APIFY_TOKEN and ANTHROPIC_API_KEY
npm run dev
```

Open http://localhost:3000.

### Demo mode (no API keys)

If `APIFY_TOKEN` or `ANTHROPIC_API_KEY` is not set, lookups fall back to mock data for four seeded profiles:

- `linkedin.com/in/priyaiyer`
- `linkedin.com/in/danielokonkwo`
- `linkedin.com/in/sorayanadim`
- `linkedin.com/in/hughpark`

These show up as suggested chips on the home screen.

### Live mode (with API keys)

Set in `.env.local`:

```
APIFY_TOKEN=apify_api_...
APIFY_LINKEDIN_PROFILE_ACTOR=dev_fusion/linkedin-profile-scraper
APIFY_LINKEDIN_POSTS_ACTOR=apimaestro/linkedin-profile-posts
ANTHROPIC_API_KEY=sk-ant-...
```

`POST /api/lookup` will:
1. Trigger the profile Actor with the target's URL.
2. Trigger the posts Actor in parallel.
3. Send both blobs + the user's profile to Claude with a strict JSON schema.
4. Return the synthesized `Person` with topics.

Find or swap actors at https://console.apify.com/actors. Override the actor IDs via the env vars above.

## Project layout

```
src/
  app/
    layout.tsx          # Inter / Source Serif 4 / JetBrains Mono fonts
    page.tsx            # Single-page router with all 7 screens
    globals.css         # Ported from prototype.css
    api/
      lookup/route.ts   # POST → Apify + Claude → Person
      feedback/route.ts # POST 👍/👎; GET list (in-memory)
  components/
    UI.tsx              # Avatar, BrandLink, AvatarMenu, Icon, Toast
    Onboarding.tsx      # 3-step first-run
    Home.tsx            # Lookup input, suggestions, recent history
    LoadingScreen.tsx   # 4-step progress while waiting on API
    Results.tsx         # Editorial topic list, ratings, downvote panel
    HistoryPage.tsx     # Searchable past lookups
    ProfilePage.tsx     # Editable user profile
    AdminPage.tsx       # Feedback queue + eval sets
  lib/
    apify.ts            # ApifyClient wrapper for profile + posts actors
    llm.ts              # Anthropic call with strict JSON output contract
    state.ts            # localStorage hook + relativeTime helper
    mockPeople.ts       # Demo data for mock mode
    types.ts            # Person, Topic, AppState, RatingState
```

## Scripts

```
npm run dev      # localhost:3000
npm run build    # production build
npm run start    # serve the build
```

## PRD alignment (v1 scope)

| PRD MVP requirement | Implementation |
|---|---|
| URL input → server-side scrape → LLM topics | `src/app/api/lookup/route.ts` |
| LinkedIn scrape via Apify | `src/lib/apify.ts` |
| Render topics with 👍/👎, qualitative downvote feedback, copy button | `src/components/Results.tsx` |
| Save lookup to history | `useAppState` (localStorage) |
| Admin dashboard for feedback review | `src/components/AdminPage.tsx` |
| Public-data-only / no protected attributes | Hard-coded in `SYSTEM_PROMPT` (`src/lib/llm.ts`) |
| <30s end-to-end | Apify calls run in parallel; loading screen exposes progress |
| 24h cache | `state.cache[handle]` short-circuits re-scrape on history click |

## Notes on the Apify integration

The app speaks directly to the Apify REST API via `apify-client`:

```ts
const run = await client.actor(PROFILE_ACTOR).call(
  { profileUrls: [url] },
  { timeout: 90, memory: 1024 }
);
const { items } = await client.dataset(run.defaultDatasetId).listItems();
```

Two actors are called in parallel — one for the profile, one for the recent posts. Both run on Apify's infra; the app polls until completion and reads from the default dataset.

Different actors expect slightly different input shapes. The wrapper accepts both `profileUrls: [url]` and `username: <handle>`/`maxPosts` to fit common community actors. If you swap in a different actor, adjust `scrapeProfile` / `scrapePosts` in `src/lib/apify.ts`.
