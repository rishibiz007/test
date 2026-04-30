import { NextRequest, NextResponse } from "next/server";
import { scrapeProfile, scrapePosts, RawProfile, RawPost } from "@/lib/apify";
import type { UserProfile } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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
    return profile.skills.slice(0, 6).join(", ");
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

    if (!rawProfile) {
      return NextResponse.json(
        { error: "Could not fetch LinkedIn profile. Check the URL and try again." },
        { status: 502 }
      );
    }

    const name = rawProfile.fullName || "";
    const userProfile: UserProfile = {
      name,
      initials: deriveInitials(name),
      role: formatRole(rawProfile),
      email: "",
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
