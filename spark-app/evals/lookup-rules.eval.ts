/**
 * Eval: LinkedIn lookup system prompt rules
 *
 * Verifies that every Claude response for a lookup satisfies all 6 rules
 * defined in the Ice Breaker system prompt.
 *
 * Run: npx braintrust eval evals/lookup-rules.eval.ts
 */

import { Eval } from "braintrust";
import { generateTopics } from "../src/lib/llm";
import type { RawProfile, RawPost } from "../src/lib/apify";
import type { UserProfile, Person } from "../src/lib/types";

// ─── Test dataset ────────────────────────────────────────────────────────────

const TARGET_PROFILE: RawProfile = {
  fullName: "Alex Rivera",
  headline: "VP of Product at Figma | ex-Notion | Building tools for thought",
  about:
    "I spend my days thinking about how software shapes the way teams work together. Before Figma I led product at Notion for 3 years. I write occasionally about product strategy and design systems.",
  currentJobTitle: "VP of Product",
  currentCompany: "Figma",
  location: "San Francisco, CA",
  experiences: [
    { title: "VP of Product", company: "Figma", duration: "2 years" },
    { title: "Head of Product", company: "Notion", duration: "3 years" },
  ],
  educations: [{ school: "Stanford University", degree: "BS Computer Science", year: "2014" }],
  skills: ["Product Strategy", "Design Systems", "Team Leadership", "B2B SaaS"],
};

const TARGET_POSTS: RawPost[] = [
  {
    text: "We just shipped multiplayer comments in Figma's new Dev Mode. Biggest insight from building it: the handoff problem isn't a file format problem — it's a communication problem. Curious how other PM teams are thinking about this.",
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
];

const USER_PROFILE: UserProfile = {
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

interface EvalInput {
  handle: string;
  profile: RawProfile;
  posts: RawPost[];
  user: UserProfile;
}

// ─── Scorers (one per rule) ───────────────────────────────────────────────────

const VALID_CATEGORIES = new Set([
  "RECENT ACTIVITY",
  "MUTUAL OVERLAP",
  "OUTSIDE WORK",
  "SHARED INTEREST",
  "WORK",
]);

const BANNED_PHRASES = [
  "bold move",
  "great post",
  "impressive",
  "love this",
  "amazing work",
  "fantastic",
  "congrats",
  "well done",
];

const PROTECTED_ATTRS = [
  "health",
  "religion",
  "religious",
  "politics",
  "political",
  "family",
  "ethnic",
  "ethnicity",
  "sexual orientation",
  "race",
  "gender",
  "disability",
  "age",
];

// Rule 1: Output contains 3–5 topics
function scoreTopicCount({ output }: { input: EvalInput; output: Person }) {
  const count = output.topics.length;
  return {
    name: "rule_topic_count_3_to_5",
    score: count >= 3 && count <= 5 ? 1 : 0,
    metadata: { topic_count: count },
  };
}

// Rule 2: Every topic has a specific source
function scoreAllTopicsHaveSource({ output }: { input: EvalInput; output: Person }) {
  const missing = output.topics.filter((t) => !t.source?.trim());
  return {
    name: "rule_all_topics_have_source",
    score: missing.length === 0 ? 1 : 0,
    metadata: { topics_missing_source: missing.map((t) => t.id) },
  };
}

// Rule 3: Every topic uses a valid category
function scoreValidCategories({ output }: { input: EvalInput; output: Person }) {
  const invalid = output.topics.filter((t) => !VALID_CATEGORIES.has(t.category));
  return {
    name: "rule_valid_categories_only",
    score: invalid.length === 0 ? 1 : 0,
    metadata: { invalid: invalid.map((t) => ({ id: t.id, category: t.category })) },
  };
}

// Rule 4: No banned generic compliment phrases
function scoreNoBannedPhrases({ output }: { input: EvalInput; output: Person }) {
  const allText = output.topics
    .map((t) => `${t.starter} ${t.why}`)
    .join(" ")
    .toLowerCase();
  const found = BANNED_PHRASES.filter((phrase) => allText.includes(phrase));
  return {
    name: "rule_no_generic_compliments",
    score: found.length === 0 ? 1 : 0,
    metadata: { banned_phrases_found: found },
  };
}

// Rule 5: No protected attributes referenced
function scoreNoProtectedAttributes({ output }: { input: EvalInput; output: Person }) {
  const allText = output.topics
    .map((t) => `${t.starter} ${t.why}`)
    .join(" ")
    .toLowerCase();
  const found = PROTECTED_ATTRS.filter((attr) => allText.includes(attr));
  return {
    name: "rule_no_protected_attributes",
    score: found.length === 0 ? 1 : 0,
    metadata: { protected_attrs_found: found },
  };
}

// Rule 6: usedYou=true only on topics that genuinely reference the user's profile
function scoreUsedYouAccuracy({ input, output }: { input: EvalInput; output: Person }) {
  const userKeywords = [
    input.user.name.toLowerCase(),
    ...input.user.education.toLowerCase().split(/[\s,]+/),
    ...input.user.recentPosts.toLowerCase().split(/\s+/),
    ...input.user.talksAbout.toLowerCase().split(/[\s,]+/),
    ...input.user.role.toLowerCase().split(/[\s,@]+/),
  ].filter((w) => w.length > 4);

  const flagged = output.topics.filter((t) => t.usedYou);
  const wronglyFlagged = flagged.filter((t) => {
    const topicText = `${t.starter} ${t.why}`.toLowerCase();
    return !userKeywords.some((kw) => topicText.includes(kw));
  });

  return {
    name: "rule_used_you_accuracy",
    score: wronglyFlagged.length === 0 ? 1 : 0,
    metadata: {
      flagged_count: flagged.length,
      wrongly_flagged: wronglyFlagged.map((t) => t.id),
    },
  };
}

// ─── Eval ─────────────────────────────────────────────────────────────────────

Eval("Ice Breaker", {
  data: (): { input: EvalInput }[] => [
    {
      input: {
        handle: "linkedin.com/in/alexrivera",
        profile: TARGET_PROFILE,
        posts: TARGET_POSTS,
        user: USER_PROFILE,
      },
    },
  ],
  task: async (input: EvalInput): Promise<Person> => generateTopics(input),
  scores: [
    scoreTopicCount,
    scoreAllTopicsHaveSource,
    scoreValidCategories,
    scoreNoBannedPhrases,
    scoreNoProtectedAttributes,
    scoreUsedYouAccuracy,
  ],
  metadata: {
    description: "Verifies all 6 system prompt rules are met for a LinkedIn lookup",
    model: "claude-sonnet-4-6",
    target: "Alex Rivera · VP Product · Figma",
  },
});
