import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { getSupabase } from "@/lib/supabase";
import { scrapeLinkedIn } from "@/lib/apify";
import { generateTopics } from "@/lib/llm";
import { MOCK_PEOPLE, DEFAULT_USER } from "@/lib/mockPeople";
import type { UserProfile } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 120;

const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

function cleanHandle(input: string): string {
  return input.trim().replace(/^https?:\/\//, "").replace(/^www\./, "").replace(/\/$/, "");
}

export async function POST(req: NextRequest) {
  let body: { handle?: string; user?: UserProfile };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const handle = cleanHandle(body.handle ?? "");
  if (!handle.includes("linkedin.com/in/")) {
    return NextResponse.json(
      { error: "That doesn't look like a LinkedIn profile URL." },
      { status: 400 }
    );
  }

  const session = await getServerSession(authOptions);
  const userId = session?.user?.id ?? null;
  const user: UserProfile = body.user ?? DEFAULT_USER;
  const supabase = getSupabase();

  const useMocks = !process.env.APIFY_TOKEN || !process.env.ANTHROPIC_API_KEY;

  if (useMocks) {
    const fallback = MOCK_PEOPLE[handle];
    if (!fallback) {
      return NextResponse.json(
        {
          error:
            "Demo mode: no mock data for this profile. Set APIFY_TOKEN and ANTHROPIC_API_KEY in .env.local for live scraping, or pick a suggested profile.",
        },
        { status: 404 }
      );
    }
    return NextResponse.json({ person: fallback, source: "mock" });
  }

  // Check server-side cache in Supabase
  if (supabase) {
    const { data: cached } = await supabase
      .from("lookup_cache")
      .select("person_json, cached_at")
      .eq("handle", handle)
      .single();

    if (cached && Date.now() - new Date(cached.cached_at as string).getTime() < CACHE_TTL_MS) {
      if (userId) {
        await supabase.from("user_lookups").insert({ user_id: userId, handle });
      }
      return NextResponse.json({ person: cached.person_json, source: "cache" });
    }
  }

  try {
    const snapshot = await scrapeLinkedIn(handle);
    console.log(`[lookup] snapshot profile=${!!snapshot.profile} posts=${snapshot.posts.length}`);
    if (!snapshot.profile && snapshot.posts.length === 0) {
      return NextResponse.json(
        { error: "Apify returned no profile data. The actor may be misconfigured or rate-limited." },
        { status: 502 }
      );
    }
    const person = await generateTopics({
      handle,
      profile: snapshot.profile,
      posts: snapshot.posts,
      user,
    });

    // Persist to shared server-side cache
    if (supabase) {
      await supabase
        .from("lookup_cache")
        .upsert({ handle, person_json: person, cached_at: new Date().toISOString() });

      if (userId) {
        await supabase.from("user_lookups").insert({ user_id: userId, handle });
      }
    }

    return NextResponse.json({ person, source: "live" });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    const isPlanError = message.toLowerCase().includes("free apify plan") || message.toLowerCase().includes("not via other methods");
    const status = isPlanError ? 402 : 500;
    console.error(`[lookup] error (${status}):`, message);
    return NextResponse.json({ error: isPlanError ? "Apify free plan does not allow API scraping. Upgrade to the Starter plan ($49/mo) at apify.com to use live LinkedIn lookups." : message }, { status });
  }
}
