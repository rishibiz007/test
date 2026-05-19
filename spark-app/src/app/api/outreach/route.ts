import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import type { Person, UserProfile } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 30;

const SYSTEM_PROMPT = `You write personalized LinkedIn connection request messages for job seekers.

Rules:
- 280 characters or fewer (LinkedIn's connection request limit)
- Reference exactly ONE specific, concrete detail from the provided ice breaker topic — something recent and non-obvious
- Briefly tie it to the sender's background or goals
- End with a clear, low-friction ask (coffee chat, 15-min call)
- No flattery: ban "huge fan", "inspiring", "love your work", "amazing"
- No vague openers: ban "pick your brain", "would love to connect", "reaching out because"
- Sound like a real person, not a template
- Output ONLY the message text. No quotes, no label, no commentary.`;

export async function POST(req: NextRequest) {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: { person: Person; user: UserProfile; topicId?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { person, user, topicId } = body;
  if (!person || !user) {
    return NextResponse.json({ error: "person and user are required" }, { status: 400 });
  }

  // Use the caller's preferred (liked) topic; fall back to first topic
  const topic = (topicId ? person.topics.find((t) => t.id === topicId) : null) ?? person.topics[0];
  if (!topic) {
    return NextResponse.json({ error: "No topics available for outreach" }, { status: 400 });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "ANTHROPIC_API_KEY is not configured" }, { status: 500 });
  }

  const userPrompt = `Target: ${person.name}, ${person.role} at ${person.company}

Ice breaker topic to reference:
- Opener: ${topic.starter}
- Context: ${topic.why}
- Source: ${topic.source}

Sender:
- Name: ${user.name}
- Role: ${user.role}
- Looking for: ${user.lookingFor || "networking and career opportunities"}

Write the LinkedIn connection request now.`;

  try {
    const client = new Anthropic({ apiKey });
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
