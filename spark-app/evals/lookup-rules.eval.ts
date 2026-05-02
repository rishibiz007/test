/**
 * Eval: LinkedIn lookup system prompt rules + RAG quality metrics
 *
 * 6 rule scorers  — deterministic checks against system prompt constraints
 * 3 LLM scorers   — Context Relevance, Groundedness, Answer Relevance (Claude Haiku judge)
 *
 * Run: npm run eval
 */

import { Eval } from "braintrust";
import Anthropic from "@anthropic-ai/sdk";
import { generateTopics } from "../src/lib/llm";
import type { RawProfile, RawPost } from "../src/lib/apify";
import type { UserProfile, Person } from "../src/lib/types";

// ─── Types ────────────────────────────────────────────────────────────────────

interface EvalInput {
  handle: string;
  profile: RawProfile;
  posts: RawPost[];
  user: UserProfile;
}

// ─── LLM judge helper (Claude Haiku) ─────────────────────────────────────────

async function judge(prompt: string): Promise<{ score: number; reasoning: string }> {
  const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });
  const res = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 400,
    messages: [{
      role: "user",
      content: `${prompt}\n\nRespond with JSON only: { "score": <0.0 to 1.0>, "reasoning": "<one sentence>" }`,
    }],
  });
  const text = (res.content[0] as { text: string }).text.trim();
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return { score: 0, reasoning: "Failed to parse judge response" };
  return JSON.parse(match[0]) as { score: number; reasoning: string };
}

function profileSummary(profile: RawProfile, posts: RawPost[]): string {
  return [
    `Name: ${profile.fullName}`,
    `Headline: ${profile.headline}`,
    `About: ${profile.about?.slice(0, 300)}`,
    `Current role: ${profile.currentJobTitle} at ${profile.currentCompany}`,
    `Skills: ${(profile.skills ?? []).slice(0, 6).join(", ")}`,
    `Recent posts:\n${posts.map((p) => `- ${p.text?.slice(0, 150)}`).join("\n")}`,
  ].join("\n");
}

function topicsSummary(topics: Person["topics"]): string {
  return topics
    .map((t, i) => `${i + 1}. [${t.category}] "${t.starter}"\n   Why: ${t.why}\n   Source: ${t.source}`)
    .join("\n");
}

// ─── Test dataset — 3 diverse LinkedIn profiles ───────────────────────────────

const SHARED_USER: UserProfile = {
  name: "Maya Patel",
  initials: "MP",
  role: "Senior PM at Linear",
  email: "",
  linkedin: "linkedin.com/in/mayapatel",
  education: "MIT, BS Computer Science, 2016",
  recentPosts: "Wrote about async collaboration and reducing meeting load for engineering teams",
  podcasts: "",
  lookingFor: "PM leadership roles at design-tool or developer-tool companies",
  talksAbout: "Product strategy, developer tools, async work, design systems",
  refreshedAt: new Date().toISOString(),
};

const TEST_CASES: { input: EvalInput }[] = [
  // Case 1: VP Product at a design tool company (strong overlap with user)
  {
    input: {
      handle: "linkedin.com/in/alexrivera",
      profile: {
        fullName: "Alex Rivera",
        headline: "VP of Product at Figma | ex-Notion | Building tools for thought",
        about: "I spend my days thinking about how software shapes the way teams work together. Before Figma I led product at Notion for 3 years. I write occasionally about product strategy and design systems.",
        currentJobTitle: "VP of Product",
        currentCompany: "Figma",
        location: "San Francisco, CA",
        experiences: [
          { title: "VP of Product", company: "Figma", duration: "2 years" },
          { title: "Head of Product", company: "Notion", duration: "3 years" },
        ],
        educations: [{ school: "Stanford University", degree: "BS Computer Science", year: "2014" }],
        skills: ["Product Strategy", "Design Systems", "Team Leadership", "B2B SaaS"],
      },
      posts: [
        {
          text: "We just shipped multiplayer comments in Figma's Dev Mode. Biggest insight: the handoff problem isn't a file format problem — it's a communication problem. Curious how other PM teams are thinking about this.",
          postedAt: "2 weeks ago",
          url: "https://linkedin.com/posts/alexrivera/devmode",
          reactionsCount: 312,
        },
        {
          text: "Hot take: most design systems fail not because of bad components but because of bad documentation culture. The component is the easy part.",
          postedAt: "1 month ago",
          url: "https://linkedin.com/posts/alexrivera/designsystems",
          reactionsCount: 198,
        },
      ],
      user: SHARED_USER,
    },
  },

  // Case 2: Engineering Manager at a developer-tool company (partial overlap)
  {
    input: {
      handle: "linkedin.com/in/jordankim",
      profile: {
        fullName: "Jordan Kim",
        headline: "Engineering Manager at Vercel | ex-GitHub | Open source contributor",
        about: "I lead the developer experience team at Vercel, focused on making deployment feel invisible. Prev GitHub Actions. I care deeply about open source sustainability and async-first engineering culture.",
        currentJobTitle: "Engineering Manager",
        currentCompany: "Vercel",
        location: "Remote (Austin, TX)",
        experiences: [
          { title: "Engineering Manager", company: "Vercel", duration: "18 months" },
          { title: "Senior Engineer", company: "GitHub", duration: "4 years" },
        ],
        educations: [{ school: "UT Austin", degree: "BS Computer Science", year: "2015" }],
        skills: ["Developer Experience", "CI/CD", "Open Source", "TypeScript", "Async Culture"],
      },
      posts: [
        {
          text: "Async-first doesn't mean no meetings. It means every meeting has a document. We cut our recurring meetings by 60% at Vercel and shipped faster. Here's the exact template we use.",
          postedAt: "1 week ago",
          url: "https://linkedin.com/posts/jordankim/async",
          reactionsCount: 541,
        },
        {
          text: "Open source maintainer burnout is a real crisis. We sponsor 12 maintainers whose packages Vercel depends on. Cost is trivial. Impact is not.",
          postedAt: "3 weeks ago",
          url: "https://linkedin.com/posts/jordankim/opensource",
          reactionsCount: 289,
        },
      ],
      user: SHARED_USER,
    },
  },

  // Case 3: Climate tech founder (minimal overlap — tests model doesn't hallucinate connections)
  {
    input: {
      handle: "linkedin.com/in/priyasharma",
      profile: {
        fullName: "Priya Sharma",
        headline: "Founder & CEO at Earthbank | Climate fintech | YC W23",
        about: "Building the financial infrastructure for the net-zero transition. Earthbank helps mid-market companies finance renewable energy projects with no upfront capital. Former energy policy analyst at the World Bank.",
        currentJobTitle: "Founder & CEO",
        currentCompany: "Earthbank",
        location: "New York, NY",
        experiences: [
          { title: "Founder & CEO", company: "Earthbank", duration: "2 years" },
          { title: "Energy Policy Analyst", company: "World Bank", duration: "3 years" },
        ],
        educations: [{ school: "Columbia University", degree: "MPP Energy Policy", year: "2018" }],
        skills: ["Climate Finance", "Renewable Energy", "Policy", "Fintech", "B2B Sales"],
      },
      posts: [
        {
          text: "We closed our $4M seed round last month. What surprised me: the hardest part wasn't finding investors — it was explaining why climate fintech is different from carbon offsets. They're not the same thing.",
          postedAt: "3 weeks ago",
          url: "https://linkedin.com/posts/priyasharma/seed",
          reactionsCount: 743,
        },
        {
          text: "Most mid-market companies want to go green but think they can't afford it. The real problem is that green finance products were designed for Fortune 500 balance sheets. We're fixing that.",
          postedAt: "2 months ago",
          url: "https://linkedin.com/posts/priyasharma/greenfinance",
          reactionsCount: 312,
        },
      ],
      user: SHARED_USER,
    },
  },
];

// ─── Rule scorers (deterministic) ─────────────────────────────────────────────

const VALID_CATEGORIES = new Set([
  "RECENT ACTIVITY", "MUTUAL OVERLAP", "OUTSIDE WORK", "SHARED INTEREST", "WORK",
]);

const BANNED_PHRASES = [
  "bold move", "great post", "impressive", "love this",
  "amazing work", "fantastic", "congrats", "well done",
];

const PROTECTED_ATTRS = [
  "health", "religion", "religious", "politics", "political",
  "family", "ethnic", "ethnicity", "sexual orientation",
  "race", "gender", "disability", "age",
];

function scoreTopicCount({ output }: { input: EvalInput; output: Person }) {
  const count = output.topics.length;
  return { name: "rule_topic_count_3_to_5", score: count >= 3 && count <= 5 ? 1 : 0, metadata: { count } };
}

function scoreAllTopicsHaveSource({ output }: { input: EvalInput; output: Person }) {
  const missing = output.topics.filter((t) => !t.source?.trim()).map((t) => t.id);
  return { name: "rule_all_topics_have_source", score: missing.length === 0 ? 1 : 0, metadata: { missing } };
}

function scoreValidCategories({ output }: { input: EvalInput; output: Person }) {
  const invalid = output.topics.filter((t) => !VALID_CATEGORIES.has(t.category));
  return { name: "rule_valid_categories_only", score: invalid.length === 0 ? 1 : 0, metadata: { invalid: invalid.map((t) => t.category) } };
}

function scoreNoBannedPhrases({ output }: { input: EvalInput; output: Person }) {
  const allText = output.topics.map((t) => `${t.starter} ${t.why}`).join(" ").toLowerCase();
  const found = BANNED_PHRASES.filter((p) => allText.includes(p));
  return { name: "rule_no_generic_compliments", score: found.length === 0 ? 1 : 0, metadata: { found } };
}

function scoreNoProtectedAttributes({ output }: { input: EvalInput; output: Person }) {
  const allText = output.topics.map((t) => `${t.starter} ${t.why}`).join(" ").toLowerCase();
  const found = PROTECTED_ATTRS.filter((a) => allText.includes(a));
  return { name: "rule_no_protected_attributes", score: found.length === 0 ? 1 : 0, metadata: { found } };
}

function scoreUsedYouAccuracy({ input, output }: { input: EvalInput; output: Person }) {
  const userKeywords = [
    input.user.name, input.user.role, input.user.education,
    input.user.recentPosts, input.user.talksAbout, input.user.lookingFor,
  ].join(" ").toLowerCase().split(/[\s,@]+/).filter((w) => w.length > 4);

  const wronglyFlagged = output.topics
    .filter((t) => t.usedYou)
    .filter((t) => {
      const text = `${t.starter} ${t.why}`.toLowerCase();
      return !userKeywords.some((kw) => text.includes(kw));
    });

  return { name: "rule_used_you_accuracy", score: wronglyFlagged.length === 0 ? 1 : 0, metadata: { wronglyFlagged: wronglyFlagged.map((t) => t.id) } };
}

// ─── LLM-as-judge scorers ─────────────────────────────────────────────────────

async function scoreContextRelevance({ input, output }: { input: EvalInput; output: Person }) {
  const { score, reasoning } = await judge(`
You are evaluating an AI networking copilot.

CONTEXT PROVIDED TO THE AI (LinkedIn profile + posts):
${profileSummary(input.profile, input.posts)}

GENERATED TOPICS:
${topicsSummary(output.topics)}

QUESTION: How well do the generated topics draw on specific, concrete details from the provided context?
- Score 1.0: Every topic references specific facts, posts, or experiences from the context
- Score 0.5: Most topics reference the context but some are generic or vague
- Score 0.0: Topics are generic and could apply to anyone — not grounded in the specific context
`);
  return { name: "context_relevance", score, metadata: { reasoning } };
}

async function scoreGroundedness({ input, output }: { input: EvalInput; output: Person }) {
  const { score, reasoning } = await judge(`
You are a fact-checker evaluating an AI networking copilot.

SOURCE DATA (LinkedIn profile + posts — treat this as ground truth):
${profileSummary(input.profile, input.posts)}

GENERATED TOPICS:
${topicsSummary(output.topics)}

QUESTION: Are all factual claims in the generated topics verifiable from the source data above?
- Score 1.0: Every claim and source reference can be verified in the source data — no invented facts
- Score 0.5: Most claims are grounded but 1-2 details seem extrapolated or slightly off
- Score 0.0: Topics contain invented facts, wrong company names, roles, or events not in the source data
`);
  return { name: "groundedness", score, metadata: { reasoning } };
}

async function scoreAnswerRelevance({ input, output }: { input: EvalInput; output: Person }) {
  const { score, reasoning } = await judge(`
You are evaluating conversation starters generated for a professional networking meeting.

PERSON BEING MET: ${input.profile.fullName} — ${input.profile.headline}
USER PREPARING FOR THE MEETING: ${input.user.name}, ${input.user.role}

GENERATED CONVERSATION STARTERS:
${topicsSummary(output.topics)}

QUESTION: How relevant and useful are these conversation starters for this specific meeting?
- Score 1.0: Starters are specific, natural, non-creepy — would genuinely help open a conversation
- Score 0.5: Starters are relevant but some feel awkward, too formal, or too on-the-nose
- Score 0.0: Starters are generic, irrelevant, or would feel strange to actually say
`);
  return { name: "answer_relevance", score, metadata: { reasoning } };
}

// ─── Eval ─────────────────────────────────────────────────────────────────────

Eval("Ice Breaker", {
  data: () => TEST_CASES,
  task: async (input: EvalInput): Promise<Person> => generateTopics(input),
  scores: [
    // Rule checks
    scoreTopicCount,
    scoreAllTopicsHaveSource,
    scoreValidCategories,
    scoreNoBannedPhrases,
    scoreNoProtectedAttributes,
    scoreUsedYouAccuracy,
    // LLM-as-judge
    scoreContextRelevance,
    scoreGroundedness,
    scoreAnswerRelevance,
  ],
  metadata: {
    description: "System prompt rules + Context Relevance, Groundedness, Answer Relevance across 3 diverse profiles",
    model: "claude-sonnet-4-6",
    judge_model: "claude-haiku-4-5-20251001",
  },
});
