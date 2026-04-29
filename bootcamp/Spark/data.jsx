// Mock data for Spark wireframes
const SPARK_USER = {
  name: "Maya Patel",
  role: "PM, ex-Notion → looking",
  email: "maya.p@example.com",
  linkedin: "linkedin.com/in/mayapatel",
  education: "UC Berkeley, B.S. Computer Science",
  recentPosts: [
    "Why I'm betting on climate tools for SMBs",
    "Notes from 6 months of unstructured time",
  ],
  podcasts: ["Lenny's Podcast (guest, S4E12)"],
  lookingFor: "PM roles at Series B+ climate / fintech startups in SF or remote",
  talksAbout: "Climate tech, Indian street food, Bay Area soccer leagues, post-Notion product taste",
  refreshed: "2 days ago",
};

const SPARK_TARGET = {
  name: "Priya Iyer",
  role: "Senior Product Manager",
  company: "Stripe",
  photo: null,
  linkedin: "linkedin.com/in/priyaiyer",
  bio: "Senior PM, Payments Infra · Stripe · ex-Square · UC Berkeley '14",
};

const SPARK_TOPICS = [
  {
    id: "t1",
    category: "RECENT ACTIVITY",
    text: "She wrote a long post last week arguing that SQL fluency is now a baseline PM skill — not a nice-to-have. She frames it as a leverage issue, not a tooling one.",
    starter: "I read your SQL post twice — the framing as leverage, not literacy, is what stuck. Did that come from a specific moment at Stripe?",
    source: "Her post on Oct 14",
    sourceUrl: "linkedin.com/posts/priyaiyer_…",
    usedYou: false,
  },
  {
    id: "t2",
    category: "MUTUAL OVERLAP",
    text: "You both went to Berkeley CS — overlapping years (you '13, her '14). You may know some of the same professors, and the program changed a lot in those two years.",
    starter: "Wait, you were Berkeley CS '14? I was '13 — did you ever take Hilfinger's 61B? It was the class everyone complained about and then quoted forever after.",
    source: "Her LinkedIn education + your profile",
    sourceUrl: "",
    usedYou: true,
  },
  {
    id: "t3",
    category: "OUTSIDE WORK",
    text: "She was on the Lenny's Podcast in March talking about how she got into PM through customer support — explicitly not through the usual MBA pipeline.",
    starter: "Your Lenny's episode landed for me — the support-to-PM path is one I think about a lot. How early did you know you wanted to leave support?",
    source: "Lenny's Podcast, March 2026",
    sourceUrl: "lennyspodcast.com/ep-priya-iyer",
    usedYou: false,
  },
  {
    id: "t4",
    category: "SHARED INTEREST",
    text: "She's posted twice in the last month about climate-tech payments rails — the unsexy plumbing, not the headlines. That's adjacent to what you said you're exploring.",
    starter: "I noticed your two posts about payments rails for climate companies. I've been pulling on the same thread from the buyer side — what's the version of this you wish more PMs understood?",
    source: "Her posts on Sep 30, Oct 8",
    sourceUrl: "",
    usedYou: true,
  },
  {
    id: "t5",
    category: "WORK",
    text: "Stripe just shipped programmable Issuing controls in beta — she's quoted in the announcement post. Worth asking what shipped vs. what was scoped down.",
    starter: "Saw your name on the Issuing controls launch post. What's the gap between the version you scoped and the version that shipped?",
    source: "Stripe blog, Oct 12",
    sourceUrl: "stripe.com/blog/issuing-controls",
    usedYou: false,
  },
];

const SPARK_HISTORY = [
  { name: "Priya Iyer", company: "Stripe", when: "2 days ago" },
  { name: "Daniel Okonkwo", company: "Watershed", when: "5 days ago" },
  { name: "Soraya Nadim", company: "Ramp", when: "1 week ago" },
  { name: "Hugh Park", company: "Anthropic", when: "2 weeks ago" },
  { name: "Lila Chen", company: "Figma", when: "3 weeks ago" },
];

const SPARK_FEEDBACK = [
  {
    id: "f1", topic: "Loved your hot take on agile being a vibe.",
    category: "RECENT ACTIVITY", rating: "down",
    tags: ["Too generic", "Wrong tone"],
    text: "Sounds AI-generated. 'Hot take' is a tell.",
    person: "Priya Iyer · Stripe",
    user: "maya.p", date: "2026-04-22",
  },
  {
    id: "f2", topic: "You both went to Berkeley CS — small overlap in years.",
    category: "MUTUAL OVERLAP", rating: "up",
    tags: [], text: "",
    person: "Priya Iyer · Stripe",
    user: "maya.p", date: "2026-04-22",
  },
  {
    id: "f3", topic: "You're both into Indian street food — try asking about her favorite chaat spot in SF.",
    category: "SHARED INTEREST", rating: "down",
    tags: ["Feels weird", "Boring"],
    text: "Walking up to a senior PM and asking about chaat is creepy. Pull from work first.",
    person: "Daniel Okonkwo · Watershed",
    user: "maya.p", date: "2026-04-23",
  },
  {
    id: "f4", topic: "Ask about the SQL post — leverage framing.",
    category: "RECENT ACTIVITY", rating: "up",
    tags: [], text: "Specific, sourced, actually useful.",
    person: "Priya Iyer · Stripe",
    user: "maya.p", date: "2026-04-22",
  },
  {
    id: "f5", topic: "She likes hiking — ask about her favorite trail.",
    category: "OUTSIDE WORK", rating: "down",
    tags: ["Too generic", "Not accurate"],
    text: "Where did 'hiking' even come from? Not in her profile.",
    person: "Soraya Nadim · Ramp",
    user: "maya.p", date: "2026-04-21",
  },
  {
    id: "f6", topic: "Compliment her recent product launch — bold move.",
    category: "WORK", rating: "down",
    tags: ["Too generic", "Boring"],
    text: "'Bold move' is filler. Be specific about WHAT shipped.",
    person: "Hugh Park · Anthropic",
    user: "maya.p", date: "2026-04-15",
  },
  {
    id: "f7", topic: "Both of you worked at Square in 2019.",
    category: "MUTUAL OVERLAP", rating: "up",
    tags: [], text: "",
    person: "Lila Chen · Figma",
    user: "maya.p", date: "2026-04-08",
  },
];

const SPARK_EVAL_SETS = [
  { id: "e1", name: "v1-baseline-failures", count: 12 },
  { id: "e2", name: "creepy-line-cases", count: 4 },
  { id: "e3", name: "too-generic", count: 8 },
];

window.SPARK_DATA = {
  USER: SPARK_USER,
  TARGET: SPARK_TARGET,
  TOPICS: SPARK_TOPICS,
  HISTORY: SPARK_HISTORY,
  FEEDBACK: SPARK_FEEDBACK,
  EVAL_SETS: SPARK_EVAL_SETS,
};
