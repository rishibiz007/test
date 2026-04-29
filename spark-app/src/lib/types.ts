export type TopicCategory =
  | "RECENT ACTIVITY"
  | "MUTUAL OVERLAP"
  | "OUTSIDE WORK"
  | "SHARED INTEREST"
  | "WORK";

export interface Topic {
  id: string;
  category: TopicCategory;
  starter: string;
  why: string;
  source: string;
  url: string;
  usedYou: boolean;
}

export interface Person {
  handle: string;
  name: string;
  initials: string;
  role: string;
  company: string;
  bio: string;
  topics: Topic[];
}

export interface UserProfile {
  name: string;
  initials: string;
  role: string;
  email: string;
  linkedin: string;
  education: string;
  recentPosts: string;
  podcasts: string;
  lookingFor: string;
  talksAbout: string;
  refreshedAt: string;
}

export interface RatingState {
  rating: "up" | "down" | null;
  reasons: string[];
  note: string;
}

export interface HistoryItem {
  handle: string;
  when: string;
}

export interface AppState {
  onboarded: boolean;
  user: UserProfile;
  ratings: Record<string, RatingState>;
  history: HistoryItem[];
  lastLookup: { handle: string; ts: string } | null;
  cache: Record<string, Person>;
}
