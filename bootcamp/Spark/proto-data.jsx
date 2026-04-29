/* global React */
// Spark prototype — data store with localStorage persistence

const LOOKUPABLE_PEOPLE = {
  "linkedin.com/in/priyaiyer": {
    handle: "linkedin.com/in/priyaiyer",
    name: "Priya Iyer",
    initials: "PI",
    role: "Senior Product Manager",
    company: "Stripe",
    bio: "Senior PM, Payments Infra · Stripe · ex-Square · UC Berkeley '14",
    topics: [
      {
        id: "p1-t1",
        category: "RECENT ACTIVITY",
        starter: "I read your SQL post twice — the framing as leverage, not literacy, is what stuck. Did that come from a specific moment at Stripe?",
        why: "She wrote a long post last week arguing that SQL fluency is now a baseline PM skill — not a nice-to-have. She frames it as a leverage issue, not a tooling one.",
        source: "Her post on Oct 14",
        url: "linkedin.com/posts/priyaiyer_sql",
        usedYou: false,
      },
      {
        id: "p1-t2",
        category: "MUTUAL OVERLAP",
        starter: "Wait, you were Berkeley CS '14? I was '13 — did you ever take Hilfinger's 61B? It was the class everyone complained about and then quoted forever after.",
        why: "You both went to Berkeley CS — overlapping years (you '13, her '14). The program changed a lot in those two years and you may know the same professors.",
        source: "Her LinkedIn education + your profile",
        url: "",
        usedYou: true,
      },
      {
        id: "p1-t3",
        category: "OUTSIDE WORK",
        starter: "Your Lenny's episode landed for me — the support-to-PM path is one I think about a lot. How early did you know you wanted to leave support?",
        why: "She was on the Lenny's Podcast in March talking about how she got into PM through customer support — explicitly not through the usual MBA pipeline.",
        source: "Lenny's Podcast, March 2026",
        url: "lennyspodcast.com/ep-priya-iyer",
        usedYou: false,
      },
      {
        id: "p1-t4",
        category: "SHARED INTEREST",
        starter: "I noticed your two posts about payments rails for climate companies. I've been pulling on the same thread from the buyer side — what's the version you wish more PMs understood?",
        why: "She's posted twice in the last month about climate-tech payments rails — the unsexy plumbing, not the headlines. That's adjacent to what you said you're exploring.",
        source: "Her posts on Sep 30, Oct 8",
        url: "",
        usedYou: true,
      },
      {
        id: "p1-t5",
        category: "WORK",
        starter: "Saw your name on the Issuing controls launch post. What's the gap between the version you scoped and the version that shipped?",
        why: "Stripe just shipped programmable Issuing controls in beta — she's quoted in the announcement. Worth asking what shipped vs. what was scoped down.",
        source: "Stripe blog, Oct 12",
        url: "stripe.com/blog/issuing-controls",
        usedYou: false,
      },
    ],
  },
  "linkedin.com/in/danielokonkwo": {
    handle: "linkedin.com/in/danielokonkwo",
    name: "Daniel Okonkwo",
    initials: "DO",
    role: "Founding Engineer",
    company: "Watershed",
    bio: "Founding engineer at Watershed · ex-Stripe Climate · MIT '12",
    topics: [
      {
        id: "p2-t1",
        category: "SHARED INTEREST",
        starter: "Your Watershed talk on supplier-emissions data quality made me think about how we handled vendor onboarding at Notion. What's the worst category of data you have to clean?",
        why: "He gave a talk last month on the messy reality of supplier emissions data at climate-tech B2Bs. You said climate tech is what you're exploring next.",
        source: "GreenBiz conference talk, Apr 2026",
        url: "greenbiz.com/watershed-talk",
        usedYou: true,
      },
      {
        id: "p2-t2",
        category: "WORK",
        starter: "I saw Watershed launched the audit-ready reports. How much of that was driven by the SEC ruling versus customers asking?",
        why: "Watershed shipped audit-ready CSRD reports two weeks ago. He led the pipeline work. Worth asking which forcing function actually moved it.",
        source: "Watershed product update, Apr 14",
        url: "watershed.com/launch/audit",
        usedYou: false,
      },
      {
        id: "p2-t3",
        category: "MUTUAL OVERLAP",
        starter: "You were at Stripe Climate when Adam was building it out — I always wondered if the 'commitment, not contribution' framing came from him or was older.",
        why: "He was on the Stripe Climate team 2020-2022. You posted about admiring how Stripe Climate framed corporate carbon spending — there's a direct lineage there.",
        source: "His LinkedIn + your blog post",
        url: "",
        usedYou: true,
      },
      {
        id: "p2-t4",
        category: "RECENT ACTIVITY",
        starter: "Your post about engineers writing PRDs got a lot of pushback in the comments. Did any of it actually change your mind?",
        why: "His post 'Engineers should write the PRD' got 800+ reactions and a heated comment thread two weeks ago. Spicy entry point.",
        source: "His LinkedIn post, Apr 12",
        url: "linkedin.com/posts/danielokonkwo_prd",
        usedYou: false,
      },
    ],
  },
  "linkedin.com/in/sorayanadim": {
    handle: "linkedin.com/in/sorayanadim",
    name: "Soraya Nadim",
    initials: "SN",
    role: "Head of Product",
    company: "Ramp",
    bio: "Head of Product, Spend Platform · Ramp · ex-Plaid · NYU Stern '11",
    topics: [
      {
        id: "p3-t1",
        category: "WORK",
        starter: "Ramp's procurement launch felt very different in tone from the original spend cards — less 'finance bro' and more measured. Was that intentional?",
        why: "Ramp announced procurement two weeks ago and the brand voice on the launch page is noticeably different from prior product launches.",
        source: "Ramp launch post, Apr 12",
        url: "ramp.com/procurement",
        usedYou: false,
      },
      {
        id: "p3-t2",
        category: "SHARED INTEREST",
        starter: "I'm curious how Ramp thinks about climate-tech buyers — there's a chunk of that customer base now and the spend patterns must be weird.",
        why: "Ramp has been quietly leaning into climate-tech as a customer segment. Lines up with what you said you're exploring.",
        source: "Ramp customer page + your profile",
        url: "",
        usedYou: true,
      },
      {
        id: "p3-t3",
        category: "RECENT ACTIVITY",
        starter: "You shared the Plaid early-days retrospective — the part about sales-led versus PLG felt unresolved. Where did Ramp end up on that?",
        why: "She reshared a long Plaid origin retrospective last week, mostly without comment, but specifically called out the sales-led vs PLG section.",
        source: "Her LinkedIn share, Apr 19",
        url: "",
        usedYou: false,
      },
    ],
  },
  "linkedin.com/in/hughpark": {
    handle: "linkedin.com/in/hughpark",
    name: "Hugh Park",
    initials: "HP",
    role: "Product Manager, Claude",
    company: "Anthropic",
    bio: "PM at Anthropic on Claude · ex-Notion · Stanford '15",
    topics: [
      {
        id: "p4-t1",
        category: "MUTUAL OVERLAP",
        starter: "We overlapped at Notion for about 8 months in 2022 — were you on the AI features team yet, or was that later?",
        why: "He was at Notion from 2021-2023, overlapping with your time there. He was on early AI integrations work.",
        source: "His LinkedIn + yours",
        url: "",
        usedYou: true,
      },
      {
        id: "p4-t2",
        category: "WORK",
        starter: "Claude's projects feature feels like it has Notion DNA — was that explicit, or did you just end up there?",
        why: "He likely worked on Projects in Claude. The mental model has obvious Notion lineage. Direct lineage to ask about.",
        source: "Anthropic product + his role",
        url: "anthropic.com/news/projects",
        usedYou: false,
      },
      {
        id: "p4-t3",
        category: "RECENT ACTIVITY",
        starter: "Your post about 'taste tax' for AI products was the best framing I've seen on it. What do you do when leadership pushes back on it as soft?",
        why: "He wrote about the 'taste tax' on AI products three weeks ago — the idea that polish is non-negotiable in this category.",
        source: "His Substack post, Apr 8",
        url: "hughpark.substack.com/taste-tax",
        usedYou: false,
      },
    ],
  },
};

const DEFAULT_USER = {
  name: "Maya Patel",
  initials: "MP",
  role: "PM, ex-Notion · looking",
  email: "maya.p@example.com",
  linkedin: "linkedin.com/in/mayapatel",
  education: "UC Berkeley, B.S. Computer Science '13",
  recentPosts: "Why I'm betting on climate tools for SMBs · Notes from 6 months of unstructured time",
  podcasts: "Lenny's Podcast (guest, S4E12)",
  lookingFor: "PM roles at Series B+ climate / fintech startups in SF or remote",
  talksAbout: "Climate tech, Indian street food, Bay Area soccer leagues, post-Notion product taste",
  refreshedAt: "2 days ago",
};

const STORAGE_KEY = "spark.proto.v1";

const initialState = () => ({
  onboarded: false,
  user: { ...DEFAULT_USER },
  // Map<topicId, { rating: 'up'|'down'|null, reasons: string[], note: string }>
  ratings: {},
  // Lookups: [{ handle, when (ISO), summary }]
  history: [],
  // Most recent (last) lookup result still cached for view
  lastLookup: null, // { handle, ts }
});

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return initialState();
    const s = JSON.parse(raw);
    return { ...initialState(), ...s };
  } catch {
    return initialState();
  }
}

function saveState(s) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); } catch {}
}

function relativeTime(iso) {
  const d = new Date(iso);
  const diffMs = Date.now() - d.getTime();
  const m = Math.round(diffMs / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const dd = Math.round(h / 24);
  if (dd === 1) return "yesterday";
  if (dd < 7)  return `${dd}d ago`;
  const w = Math.round(dd / 7);
  return `${w}w ago`;
}

window.SparkData = {
  PEOPLE: LOOKUPABLE_PEOPLE,
  DEFAULT_USER,
  initialState, loadState, saveState, relativeTime, STORAGE_KEY,
};
