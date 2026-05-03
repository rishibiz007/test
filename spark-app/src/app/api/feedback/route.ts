import { NextRequest, NextResponse } from "next/server";
import { initLogger } from "braintrust";

export const runtime = "nodejs";

interface FeedbackBody {
  topicId: string;
  rating: "up" | "down";
  reasons?: string[];
  note?: string;
  handle?: string;
  spanId?: string;
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

  if (process.env.BRAINTRUST_API_KEY && body.spanId) {
    const logger = initLogger({ projectName: "Ice Breaker", apiKey: process.env.BRAINTRUST_API_KEY });
    const metadata: Record<string, unknown> = {};
    if (body.reasons?.length) metadata.reasons = body.reasons;
    if (body.topicId) metadata.topicId = body.topicId;
    logger.logFeedback({
      id: body.spanId,
      scores: { "IB Feedback": body.rating === "up" ? 1 : 0 },
      comment: body.note || undefined,
      ...(Object.keys(metadata).length ? { metadata } : {}),
    });
  }

  return NextResponse.json({ ok: true, count: inMemoryStore.length });
}

export async function GET() {
  return NextResponse.json({ feedback: inMemoryStore });
}
