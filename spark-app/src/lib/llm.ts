import Anthropic from "@anthropic-ai/sdk";
import { initLogger, wrapAnthropic } from "braintrust";
import type { Person, Topic, UserProfile } from "./types";
import type { RawPost, RawProfile } from "./apify";

const MODEL = "claude-sonnet-4-6";

export const btLogger = process.env.BRAINTRUST_API_KEY
  ? initLogger({ projectName: "Ice Breaker", apiKey: process.env.BRAINTRUST_API_KEY })
  : null;

const SYSTEM_PROMPT = `You are Ice Breaker, a networking copilot. Given (a) a target person's public LinkedIn profile + recent posts, and (b) the user's own profile, produce 3-5 personal, timely, non-creepy talking points the user can actually open with.

Rules:
- Use ONLY public information. Never invent facts.
- Never reference protected attributes (health, religion, politics, family, ethnicity, age, sexual orientation).
- Each topic must have a specific source. If the source is missing, drop the topic.
- Prefer specificity over flattery. "Bold move" / "great post" / generic compliments are banned.
- Categories: RECENT ACTIVITY, MUTUAL OVERLAP, OUTSIDE WORK, SHARED INTEREST, WORK.
- Mark usedYou=true if the topic explicitly draws on something from the user's profile.
- Output ONLY valid JSON matching the schema. No prose, no markdown.

Schema:
{
  "name": string,
  "role": string,
  "company": string,
  "bio": string,
  "topics": [
    {
      "id": string,
      "category": "RECENT ACTIVITY" | "MUTUAL OVERLAP" | "OUTSIDE WORK" | "SHARED INTEREST" | "WORK",
      "starter": string (the actual opener — first-person, conversational, ends with a real question),
      "why": string (1-2 sentences explaining the connection — what it's drawn from),
      "source": string (human-readable source label like "Her LinkedIn post on Apr 14"),
      "url": string (URL if available, else ""),
      "usedYou": boolean
    }
  ]
}`;

function userPrompt(opts: {
  handle: string;
  profile: RawProfile | null;
  posts: RawPost[];
  user: UserProfile;
}): string {
  return `TARGET HANDLE: ${opts.handle}

TARGET PROFILE (from LinkedIn):
${JSON.stringify(opts.profile ?? {}, null, 2)}

TARGET RECENT POSTS:
${JSON.stringify(opts.posts ?? [], null, 2)}

USER PROFILE (the person preparing for the chat — Ice Breaker should personalize):
- Name: ${opts.user.name}
- Role: ${opts.user.role}
- Education: ${opts.user.education}
- Recent posts: ${opts.user.recentPosts}
- Podcasts/talks: ${opts.user.podcasts}
- Looking for: ${opts.user.lookingFor}
- Loves talking about: ${opts.user.talksAbout}

Return ONLY the JSON object. 3-5 topics. No commentary.`;
}

function deriveInitials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

export async function generateTopics(opts: {
  handle: string;
  profile: RawProfile | null;
  posts: RawPost[];
  user: UserProfile;
}): Promise<Person> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY is not configured. Add it to .env.local.");
  }
  const baseClient = new Anthropic({ apiKey });
  const client = btLogger ? wrapAnthropic(baseClient) : baseClient;

  const build = async (spanId?: string): Promise<Person> => {
    const message = await client.messages.create({
      model: MODEL,
      max_tokens: 2048,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userPrompt(opts) }],
    });

    const block = message.content.find((c) => c.type === "text");
    const text = block && "text" in block ? block.text.trim() : "";

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error("LLM returned non-JSON output");
    }

    const parsed = JSON.parse(jsonMatch[0]) as Partial<Person>;

    const name =
      parsed.name ||
      opts.profile?.fullName ||
      opts.handle.replace(/.*\/in\//, "").replace(/-/g, " ");
    const role = parsed.role || opts.profile?.currentJobTitle || opts.profile?.headline || "";
    const company = parsed.company || opts.profile?.currentCompany || "";
    const bio = parsed.bio || opts.profile?.headline || `${role}${company ? ` · ${company}` : ""}`;

    const topics: Topic[] = (parsed.topics ?? []).map((t, i) => ({
      id: t.id ?? `t-${i + 1}`,
      category: (t.category as Topic["category"]) ?? "WORK",
      starter: t.starter ?? "",
      why: t.why ?? "",
      source: t.source ?? "",
      url: t.url ?? "",
      usedYou: !!t.usedYou,
    }));

    return {
      handle: opts.handle,
      name,
      initials: deriveInitials(name),
      role,
      company,
      bio,
      topics,
      spanId,
    };
  };

  if (!btLogger) return build();

  return btLogger.traced((span) => build(span.id), { name: "ice-breaker-lookup" });
}
