import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { scrapeProfile, scrapePosts, RawProfile, RawPost } from "@/lib/apify";
import type { UserProfile } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 120;

function cleanHandle(input: string): string {
  return input.trim().replace(/^https?:\/\//, "").replace(/^www\./, "").replace(/\/$/, "");
}

function deriveInitials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

function formatEducation(profile: RawProfile): string {
  const edu = profile.educations?.[0];
  if (!edu) return "";
  const parts = [edu.school, edu.degree, edu.year].filter(Boolean);
  return parts.join(", ");
}

function formatRole(profile: RawProfile): string {
  if (profile.currentJobTitle && profile.currentCompany) {
    return `${profile.currentJobTitle} · ${profile.currentCompany}`;
  }
  return profile.currentJobTitle || profile.currentCompany || profile.headline || "";
}

function formatRecentPosts(posts: RawPost[]): string {
  const snippets = posts
    .slice(0, 3)
    .map((p) => p.text?.split("\n")[0]?.slice(0, 80).trim())
    .filter(Boolean) as string[];
  return snippets.join(" · ");
}

function formatTalksAbout(profile: RawProfile): string {
  if (profile.skills && profile.skills.length > 0) {
    const names = profile.skills
      .map((s) => (typeof s === "string" ? s : (s as Record<string, unknown>)?.name ?? ""))
      .filter((s): s is string => typeof s === "string" && s.length > 0)
      .slice(0, 6);
    if (names.length > 0) return names.join(", ");
  }
  if (profile.about) {
    return profile.about.slice(0, 200).trim();
  }
  return "";
}

export async function POST(req: NextRequest) {
  let body: { handle?: string };
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

  const useMocks = !process.env.APIFY_TOKEN || !process.env.ANTHROPIC_API_KEY;
  if (useMocks) {
    return NextResponse.json(
      { error: "Demo mode: Set APIFY_TOKEN and ANTHROPIC_API_KEY to fetch real profiles." },
      { status: 404 }
    );
  }

  try {
    const [profile, posts] = await Promise.allSettled([
      scrapeProfile(handle),
      scrapePosts(handle, 5),
    ]);

    const rawProfile: RawProfile | null =
      profile.status === "fulfilled" ? profile.value : null;
    const rawPosts: RawPost[] =
      posts.status === "fulfilled" ? posts.value : [];

    if (profile.status === "rejected") {
      const reason = profile.reason instanceof Error ? profile.reason.message : String(profile.reason);
      console.error("[profile] scrapeProfile rejected:", reason);
      return NextResponse.json(
        { error: `LinkedIn scrape failed: ${reason}` },
        { status: 502 }
      );
    }

    if (!rawProfile) {
      console.error("[profile] scrapeProfile returned empty dataset for handle:", handle);
      return NextResponse.json(
        { error: "LinkedIn returned no data for that profile. It may be private, or LinkedIn is rate-limiting this URL. Try again in a few minutes." },
        { status: 502 }
      );
    }

    const name = rawProfile.fullName || "";
    const userProfile: UserProfile = {
      name,
      initials: deriveInitials(name),
      role: formatRole(rawProfile),
      email: (await getServerSession(authOptions))?.user?.email ?? "",
      linkedin: handle,
      education: formatEducation(rawProfile),
      recentPosts: formatRecentPosts(rawPosts),
      podcasts: "",
      lookingFor: "",
      talksAbout: formatTalksAbout(rawProfile),
      refreshedAt: new Date().toISOString(),
    };

    return NextResponse.json({ user: userProfile });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("[profile] error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
