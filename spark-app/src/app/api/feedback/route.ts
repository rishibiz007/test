import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

interface FeedbackBody {
  topicId: string;
  rating: "up" | "down";
  reasons?: string[];
  note?: string;
  handle?: string;
}

const inMemoryStore: FeedbackBody[] = [];

export async function POST(req: NextRequest) {
  let body: FeedbackBody;
  try {
    body = (await req.json()) as FeedbackBody;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }
  if (!body.topicId || !body.rating) {
    return NextResponse.json({ error: "topicId and rating are required" }, { status: 400 });
  }
  inMemoryStore.push({ ...body });
  return NextResponse.json({ ok: true, count: inMemoryStore.length });
}

export async function GET() {
  return NextResponse.json({ feedback: inMemoryStore });
}
