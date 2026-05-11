import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import type { Person, UserProfile } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 30;

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const SYSTEM_PROMPT = `You write personalized LinkedIn connection request messages for job seekers.

Rules:
- 280 characters or fewer (LinkedIn's connection request limit)
- Reference exactly ONE specific, concrete detail from the provided ice breaker topics — something recent and non-obvious
- Briefly tie it to the sender's background or goals
- End with a clear, low-friction ask (coffee chat, 15-min call)
- No flattery: ban "huge fan", "inspiring", "love your work", "amazing"
- No vague openers: ban "pick your brain", "would love to connect", "reaching out because"
- Sound like a real person, not a template
- Output ONLY the message text. No quotes, no label, no commentary.`;

export async function POST(req: NextRequest) {
  let body: { person: Person; user: UserProfile };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { person, user } = body;
  if (!person || !user) {
    return NextResponse.json({ error: "person and user are required" }, { status: 400 });
  }

  const bestTopic = person.topics[0];
  if (!bestTopic) {
    return NextResponse.json({ error: "No topics available for outreach" }, { status: 400 });
  }

  const userPrompt = `Target: ${person.name}, ${person.role} at ${person.company}

Best ice breaker topic to reference:
- Opener: ${bestTopic.starter}
- Context: ${bestTopic.why}
- Source: ${bestTopic.source}

Sender:
- Name: ${user.name}
- Role: ${user.role}
- Looking for: ${user.lookingFor || "networking and career opportunities"}

Write the LinkedIn connection request now.`;

  try {
    const message = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userPrompt }],
    });

    const block = message.content.find((c) => c.type === "text");
    const text = block && "text" in block ? block.text.trim() : "";

    return NextResponse.json({ message: text });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Unknown error";
    console.error("[outreach] error:", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
