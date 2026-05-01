import * as amplitude from "@amplitude/analytics-browser";

let initialized = false;

export function initAmplitude() {
  if (initialized || typeof window === "undefined") return;
  const key = process.env.NEXT_PUBLIC_AMPLITUDE_API_KEY;
  if (!key) return;
  amplitude.init(key, { defaultTracking: false });
  initialized = true;
}

// Onboarding
export function trackOnboardingLinkedInSubmitted(linkedinUrl: string) {
  amplitude.track("onboarding_linkedin_submitted", { linkedin_url: linkedinUrl });
}

export function trackOnboardingCompleted(linkedinUrl: string, name: string) {
  amplitude.track("onboarding_completed", { linkedin_url: linkedinUrl, name });
}

// Lookup
export function trackLookupSubmitted(linkedinUrl: string) {
  amplitude.track("lookup_submitted", { linkedin_url: linkedinUrl });
}

export function trackLookupResponseReceived(props: {
  target_handle: string;
  target_name: string;
  topic_count: number;
  source: string;
}) {
  amplitude.track("lookup_response_received", props);
}

export function trackLookupFailed(linkedinUrl: string, error: string) {
  amplitude.track("lookup_failed", { linkedin_url: linkedinUrl, error });
}

// Topic feedback
export function trackThumbsUp(props: {
  topic_id: string;
  topic_category: string;
  target_handle: string;
  is_personalized: boolean;
}) {
  amplitude.track("topic_thumbs_up", props);
}

export function trackThumbsDownSubmitted(props: {
  topic_id: string;
  topic_category: string;
  target_handle: string;
  feedback: string;
  is_personalized: boolean;
}) {
  amplitude.track("topic_thumbs_down_submitted", props);
}

export function trackTopicCopied(props: {
  topic_id: string;
  topic_category: string;
  target_handle: string;
}) {
  amplitude.track("topic_copied", props);
}
