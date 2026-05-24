import { NextRequest, NextResponse } from "next/server";
import { MindStudioAgent } from "@mindstudio-ai/agent";
import type { Person, UserProfile } from "@/lib/types";

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

  // Run TTS first, then email the audio link
  const { audioUrl } = await agent.textToSpeech({ text: script });

  if (user.email) {
    const firstName = person.name.split(" ")[0];
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
  }

  return NextResponse.json({ audioUrl, script });
}
