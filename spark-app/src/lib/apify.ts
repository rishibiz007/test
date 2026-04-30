export interface RawProfile {
  fullName?: string;
  headline?: string;
  about?: string;
  currentCompany?: string;
  currentJobTitle?: string;
  location?: string;
  experiences?: Array<{ title?: string; company?: string; duration?: string; description?: string }>;
  educations?: Array<{ school?: string; degree?: string; year?: string }>;
  skills?: (string | { name?: string })[];
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

export interface ScrapedSnapshot {
  profile: RawProfile | null;
  posts: RawPost[];
}

const APIFY_BASE = "https://api.apify.com/v2";

const PROFILE_ACTOR =
  process.env.APIFY_LINKEDIN_PROFILE_ACTOR || "dev_fusion/linkedin-profile-scraper";
const POSTS_ACTOR =
  process.env.APIFY_LINKEDIN_POSTS_ACTOR || "apimaestro/linkedin-profile-posts";

function token(): string {
  const t = process.env.APIFY_TOKEN;
  if (!t) throw new Error("APIFY_TOKEN is not configured.");
  return t;
}

function normalizeUrl(handle: string): string {
  const trimmed = handle.trim().replace(/\/$/, "");
  if (trimmed.startsWith("http")) return trimmed;
  return `https://www.${trimmed.replace(/^www\./, "")}`;
}

function extractUsername(handle: string): string {
  const m = handle.match(/in\/([^\/?#]+)/);
  return m ? m[1] : handle;
}

async function runActor(actorId: string, input: Record<string, unknown>, timeoutSecs = 90): Promise<string> {
  const url = `${APIFY_BASE}/acts/${encodeURIComponent(actorId)}/runs?token=${token()}&timeout=${timeoutSecs}&waitForFinish=${timeoutSecs}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Apify run failed (${res.status}): ${text}`);
  }
  const json = await res.json() as { data: { defaultDatasetId: string; status: string } };
  console.log(`[apify] actor=${actorId} status=${json.data.status} dataset=${json.data.defaultDatasetId}`);
  return json.data.defaultDatasetId;
}

async function getDatasetItems<T>(datasetId: string): Promise<T[]> {
  const url = `${APIFY_BASE}/datasets/${datasetId}/items?token=${token()}&clean=true`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Apify dataset fetch failed (${res.status})`);
  }
  return res.json() as Promise<T[]>;
}

export async function scrapeProfile(handle: string): Promise<RawProfile | null> {
  const url = normalizeUrl(handle);
  console.log(`[apify] scrapeProfile → actor=${PROFILE_ACTOR} url=${url}`);
  const datasetId = await runActor(PROFILE_ACTOR, { profileUrls: [url] });
  const items = await getDatasetItems<RawProfile>(datasetId);
  console.log(`[apify] scrapeProfile items=${items.length}`);
  const first = items[0] as Record<string, unknown> | undefined;
  if (first && typeof first["error"] === "string") {
    throw new Error(`Apify actor error (${PROFILE_ACTOR}): ${first["error"]}`);
  }
  return items[0] ?? null;
}

export async function scrapePosts(handle: string, limit = 8): Promise<RawPost[]> {
  const url = normalizeUrl(handle);
  const username = extractUsername(handle);
  console.log(`[apify] scrapePosts → actor=${POSTS_ACTOR} url=${url} username=${username}`);
  const datasetId = await runActor(POSTS_ACTOR, { username, profileUrls: [url], maxPosts: limit, limit });
  const items = await getDatasetItems<RawPost>(datasetId);
  console.log(`[apify] scrapePosts items=${items.length}`);
  return items.slice(0, limit);
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
