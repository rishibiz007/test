import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { getSupabase } from "@/lib/supabase";
import type { Person } from "@/lib/types";

export const runtime = "nodejs";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ history: [], cache: {} });
  }

  const supabase = getSupabase();
  if (!supabase) {
    return NextResponse.json({ history: [], cache: {} });
  }

  const { data: lookups } = await supabase
    .from("user_lookups")
    .select("handle, looked_up_at")
    .eq("user_id", session.user.id)
    .order("looked_up_at", { ascending: false })
    .limit(100);

  if (!lookups?.length) {
    return NextResponse.json({ history: [], cache: {} });
  }

  // Dedupe by handle, keep most recent timestamp per handle
  const seen = new Set<string>();
  const unique = lookups.filter((l) => {
    if (seen.has(l.handle)) return false;
    seen.add(l.handle);
    return true;
  }).slice(0, 50);

  const handles = unique.map((l) => l.handle);

  const { data: cached } = await supabase
    .from("lookup_cache")
    .select("handle, person_json")
    .in("handle", handles);

  const cacheMap: Record<string, Person> = {};
  for (const item of cached ?? []) {
    cacheMap[item.handle] = item.person_json as Person;
  }

  return NextResponse.json({
    history: unique.map((l) => ({ handle: l.handle, when: l.looked_up_at })),
    cache: cacheMap,
  });
}
