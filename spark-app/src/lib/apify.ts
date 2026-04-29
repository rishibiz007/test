import { ApifyClient } from "apify-client";

const APIFY_TOKEN = process.env.APIFY_TOKEN;

const PROFILE_ACTOR =
  process.env.APIFY_LINKEDIN_PROFILE_ACTOR || "dev_fusion/linkedin-profile-scraper";
const POSTS_ACTOR =
  process.env.APIFY_LINKEDIN_POSTS_ACTOR || "apimaestro/linkedin-profile-posts";

export interface RawProfile {
  fullName?: string;
  headline?: string;
  about?: string;
  currentCompany?: string;
  currentJobTitle?: string;
  location?: string;
  experiences?: Array<{ title?: string; company?: string; duration?: string; description?: string }>;
  educations?: Array<{ school?: string; degree?: string; year?: string }>;
  skills?: string[];
  publicIdentifier?: string;
  linkedinUrl?: string;
}

export interface RawPost {
  text?: string;
  postedAt?: string;
  url?: string;
  reactionsCount?: number;
  commentsCount?: number;
}

function getClient(): ApifyClient {
  if (!APIFY_TOKEN) {
    throw new Error("APIFY_TOKEN is not configured. Add it to .env.local.");
  }
  return new ApifyClient({ token: APIFY_TOKEN });
}

function normalizeUrl(handle: string): string {
  const trimmed = handle.trim().replace(/\/$/, "");
  if (trimmed.startsWith("http")) return trimmed;
  return `https://www.${trimmed.replace(/^www\./, "")}`;
}

function checkActorError(items: Record<string, unknown>[], actor: string) {
  const first = items[0] as Record<string, unknown> | undefined;
  if (first && typeof first["error"] === "string") {
    console.error(`[apify] actor=${actor} returned error item:`, first["error"]);
    throw new Error(`Apify actor error (${actor}): ${first["error"]}`);
  }
}

export async function scrapeProfile(handle: string): Promise<RawProfile | null> {
  const client = getClient();
  const url = normalizeUrl(handle);

  console.log(`[apify] scrapeProfile → actor=${PROFILE_ACTOR} url=${url}`);
  const run = await client.actor(PROFILE_ACTOR).call(
    { profileUrls: [url] },
    { timeout: 90, memory: 1024 }
  );
  console.log(`[apify] scrapeProfile run id=${run.id} status=${run.status} dataset=${run.defaultDatasetId}`);

  const { items } = await client.dataset(run.defaultDatasetId).listItems();
  console.log(`[apify] scrapeProfile items count=${items.length}`, items[0] ?? "(empty)");
  checkActorError(items as Record<string, unknown>[], PROFILE_ACTOR);
  return (items[0] as RawProfile) || null;
}

export async function scrapePosts(handle: string, limit = 8): Promise<RawPost[]> {
  const client = getClient();
  const url = normalizeUrl(handle);
  const username = extractUsername(handle);

  console.log(`[apify] scrapePosts → actor=${POSTS_ACTOR} url=${url} username=${username}`);
  const run = await client.actor(POSTS_ACTOR).call(
    { username, profileUrls: [url], maxPosts: limit, limit },
    { timeout: 90, memory: 1024 }
  );
  console.log(`[apify] scrapePosts run id=${run.id} status=${run.status} dataset=${run.defaultDatasetId}`);

  const { items } = await client.dataset(run.defaultDatasetId).listItems();
  console.log(`[apify] scrapePosts items count=${items.length}`, items[0] ?? "(empty)");
  checkActorError(items as Record<string, unknown>[], POSTS_ACTOR);
  return (items as RawPost[]).slice(0, limit);
}

function extractUsername(handle: string): string {
  const m = handle.match(/in\/([^\/?#]+)/);
  return m ? m[1] : handle;
}

export interface ScrapedSnapshot {
  profile: RawProfile | null;
  posts: RawPost[];
}

export async function scrapeLinkedIn(handle: string): Promise<ScrapedSnapshot> {
  const [profile, posts] = await Promise.allSettled([
    scrapeProfile(handle),
    scrapePosts(handle, 8),
  ]);
  if (profile.status === "rejected") {
    console.error("[apify] scrapeProfile rejected:", profile.reason);
    throw profile.reason instanceof Error ? profile.reason : new Error(String(profile.reason));
  }
  if (posts.status === "rejected") {
    console.error("[apify] scrapePosts rejected:", posts.reason);
  }
  return {
    profile: profile.value,
    posts: posts.status === "fulfilled" ? posts.value : [],
  };
}
