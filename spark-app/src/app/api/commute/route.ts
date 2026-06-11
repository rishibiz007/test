import { NextRequest, NextResponse } from "next/server";
import { MindStudioAgent } from "@mindstudio-ai/agent";
import type { Person, UserProfile } from "@/lib/types";

function describeError(err: unknown): string {
  if (err && typeof err === "object" && "code" in err) {
    const e = err as { message?: string; code?: string; status?: number; details?: unknown };
    const detail = e.details ? JSON.stringify(e.details).slice(0, 600) : "";
    const status = e.status ? ` (HTTP ${e.status})` : "";
    return `[${e.code ?? "unknown"}]${status} ${e.message ?? ""}${detail ? ` — details: ${detail}` : ""}`.trim();
  }
  return err instanceof Error ? err.message : String(err);
}

function buildScript(person: Person, user: UserProfile): string {
  const userFirst = user.name.split(" ")[0];
  const lines: string[] = [
    `Hey ${userFirst}. You're heading to meet ${person.name}, ${person.role} at ${person.company}.`,
    `Here ${person.topics.length === 1 ? "is" : "are"} ${person.topics.length} ice breaker${person.topics.length === 1 ? "" : "s"} to kick things off.`,
    "",
  ];

  person.topics.forEach((t, i) => {
    lines.push(`Ice breaker ${i + 1}.`);
    lines.push(t.starter);
    lines.push(`Why this works — ${t.why}`);
    lines.push("");
  });

  lines.push(`You've got this, ${userFirst}. Good luck with your coffee chat.`);
  return lines.join("\n");
}

export async function POST(req: NextRequest) {
  const body = await req.json() as { person: Person; user: UserProfile };
  const { person, user } = body;

  if (!person?.topics?.length) {
    return NextResponse.json({ error: "No topics to convert." }, { status: 400 });
  }

  const apiKey = process.env.MINDSTUDIO_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "MindStudio API key not configured." }, { status: 500 });
  }

  const agent = new MindStudioAgent({ apiKey });
  const script = buildScript(person, user);

  let audioUrl: string;
  try {
    const tts = await agent.textToSpeech({ text: script });
    audioUrl = tts.audioUrl;
  } catch (err) {
    const message = describeError(err);
    console.error("[commute] TTS failed:", message);
    return NextResponse.json(
      { error: `Audio generation failed: ${message}`, stage: "tts" },
      { status: 502 },
    );
  }

  let emailWarning: string | undefined;
  if (user.email) {
    try {
      await agent.sendEmail({
        to: user.email,
        subject: `Your Ice Breakers for ${person.name} — listen on the go`,
        body: [
          `Hi ${user.name.split(" ")[0]},`,
          "",
          `Here's your audio briefing for your upcoming coffee chat with ${person.name} (${person.role} at ${person.company}).`,
          "",
          `Listen here: ${audioUrl}`,
          "",
          `Ice Breakers covered:`,
          ...person.topics.map((t, i) => `${i + 1}. ${t.starter}`),
          "",
          "— Ice Breaker",
        ].join("\n"),
      });
    } catch (err) {
      const message = describeError(err);
      console.error("[commute] email send failed:", message);
      emailWarning = `Audio generated, but email failed: ${message}`;
    }
  }

  return NextResponse.json({ audioUrl, script, emailWarning });
}
