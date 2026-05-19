"use client";
import { useState } from "react";
import { Avatar, Icon } from "./UI";
import { relativeTime } from "@/lib/state";
import type { AppState, Person, RatingState, UserProfile } from "@/lib/types";
import { trackThumbsUp, trackThumbsDownSubmitted, trackTopicCopied } from "@/lib/analytics";

const DOWNVOTE_REASONS = [
  "Too generic",
  "Wrong about them",
  "Awkward / pushy",
  "Already knew this",
  "Tone is off",
  "Not relevant to me",
];

type OutreachStatus = "idle" | "loading" | "done" | "error";

interface Props {
  state: AppState;
  update: (patch: Partial<AppState>) => void;
  person: Person;
  onBackHome: () => void;
  pushToast: (t: { text: string; actionLabel?: string | null; onAction?: (() => void) | null; ttl?: number }) => void;
}

export default function Results({ state, update, person, onBackHome, pushToast }: Props) {
  const [openPanel, setOpenPanel] = useState<string | null>(null);
  const [removing, setRemoving] = useState<Record<string, boolean>>({});
  const [outreachStatus, setOutreachStatus] = useState<OutreachStatus>("idle");
  const [outreachMessage, setOutreachMessage] = useState("");

  const generateOutreach = async () => {
    setOutreachStatus("loading");
    // Prefer the first 👍'd topic so the message reflects what the user found useful
    const likedTopicId = person.topics.find((t) => state.ratings[t.id]?.rating === "up")?.id;
    try {
      const res = await fetch("/api/outreach", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ person, user: state.user, topicId: likedTopicId }),
      });
      const data = await res.json() as { message?: string; error?: string };
      if (!res.ok || !data.message) {
        setOutreachStatus("error");
        return;
      }
      setOutreachMessage(data.message);
      setOutreachStatus("done");
    } catch {
      setOutreachStatus("error");
    }
  };

  const visibleTopics = person.topics.filter((t) => {
    const r = state.ratings[t.id];
    if (r?.rating === "down" && r.reasons && r.reasons.length > 0) return false;
    return !removing[t.id];
  });

  const updateRating = (topicId: string, next: RatingState) => {
    update({ ratings: { ...state.ratings, [topicId]: next } });
  };

  const setRating = (topicId: string, rating: "up" | "down") => {
    const prev: RatingState = state.ratings[topicId] || { rating: null, reasons: [], note: "" };
    if (prev.rating === rating) {
      updateRating(topicId, { ...prev, rating: null });
      return;
    }
    if (rating === "down") {
      setOpenPanel(topicId);
      updateRating(topicId, { ...prev, rating: "down", reasons: prev.reasons || [], note: prev.note || "" });
      return;
    }
    updateRating(topicId, { rating, reasons: [], note: "" });
    const topic = person.topics.find((t) => t.id === topicId);
    trackThumbsUp({
      topic_id: topicId,
      topic_category: topic?.category ?? "",
      target_handle: person.handle,
      is_personalized: topic?.usedYou ?? false,
    });
    void fetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ topicId, rating, handle: person.handle, spanId: person.spanId }),
    }).catch(() => {});
    pushToast({
      text: "Thanks — Ice Breaker will surface more like this.",
      actionLabel: "Undo",
      onAction: () => updateRating(topicId, prev),
    });
  };

  const submitDownvote = (topicId: string) => {
    const cur = state.ratings[topicId];
    if (!cur || ((!cur.reasons || cur.reasons.length === 0) && !cur.note?.trim())) return;
    const topic = person.topics.find((t) => t.id === topicId);
    const feedbackParts = [...(cur.reasons || []), ...(cur.note?.trim() ? [cur.note.trim()] : [])];
    trackThumbsDownSubmitted({
      topic_id: topicId,
      topic_category: topic?.category ?? "",
      target_handle: person.handle,
      feedback: feedbackParts.join(", "),
      is_personalized: topic?.usedYou ?? false,
    });
    setOpenPanel(null);
    setRemoving((p) => ({ ...p, [topicId]: true }));
    void fetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        topicId,
        rating: "down",
        reasons: cur.reasons,
        note: cur.note,
        handle: person.handle,
        spanId: person.spanId,
      }),
    }).catch(() => {});
    pushToast({
      text: "Removed from this list. Ice Breaker won't suggest things like this.",
      actionLabel: "Undo",
      onAction: () => {
        const next = { ...state.ratings };
        delete next[topicId];
        update({ ratings: next });
        setRemoving((p) => {
          const n = { ...p };
          delete n[topicId];
          return n;
        });
      },
    });
  };

  const cancelDownvote = (topicId: string) => {
    const next = { ...state.ratings };
    delete next[topicId];
    update({ ratings: next });
    setOpenPanel(null);
  };

  const toggleReason = (topicId: string, reason: string) => {
    const cur: RatingState =
      state.ratings[topicId] || { rating: "down", reasons: [], note: "" };
    const reasons = cur.reasons.includes(reason)
      ? cur.reasons.filter((r) => r !== reason)
      : [...cur.reasons, reason];
    updateRating(topicId, { ...cur, reasons });
  };

  const setNote = (topicId: string, note: string) => {
    const cur: RatingState =
      state.ratings[topicId] || { rating: "down", reasons: [], note: "" };
    updateRating(topicId, { ...cur, note });
  };

  return (
    <div className="page fade-in" data-screen-label="06 results">
      <button
        className="btn ghost sm"
        onClick={onBackHome}
        style={{ marginLeft: -10, marginBottom: 14 }}
      >
        ← New lookup
      </button>

      <div className="editorial-meta">
        <Avatar initials={person.initials} size="xl" />
        <div>
          <div className="eyebrow" style={{ marginBottom: 6 }}>
            SPARK BRIEF · {person.topics.length} TOPICS
          </div>
          <h1 className="h-display serif" style={{ marginBottom: 6 }}>
            {person.name}
          </h1>
          <div className="muted" style={{ fontSize: 14 }}>
            {person.bio}
          </div>
        </div>
        <span className="badge success" title="Cached for 24 hours">
          CACHED
        </span>
      </div>

      <div className="stagger" style={{ position: "relative" }}>
        {visibleTopics.length === 0 ? (
          <div className="empty">
            <div className="em-title">All topics dismissed</div>
            <div style={{ fontSize: 13 }}>
              Ice Breaker learned from those signals. Try another lookup.
            </div>
            <button className="btn secondary sm" style={{ marginTop: 14 }} onClick={onBackHome}>
              Look someone else up →
            </button>
          </div>
        ) : (
          visibleTopics.map((t, i) => {
            const r = state.ratings[t.id] || { rating: null, reasons: [], note: "" };
            const liked = r.rating === "up";
            const downOpen = openPanel === t.id;
            const isRemoving = removing[t.id];
            return (
              <article
                key={t.id}
                className={`topic ${liked ? "liked" : ""} ${isRemoving ? "removing" : ""}`}
                style={{ marginLeft: 0 }}
              >
                <div className="topic-num">{String(i + 1).padStart(2, "0")}</div>
                <div className="topic-head">
                  <span className="tag">{t.category}</span>
                  {t.usedYou && (
                    <span className="badge" title="Personalized using your profile">
                      YOUR PROFILE
                    </span>
                  )}
                  <div className="topic-actions">
                    <button
                      className={`icon-btn ${liked ? "active" : ""}`}
                      onClick={() => setRating(t.id, "up")}
                      title="Helpful"
                      aria-pressed={liked}
                    >
                      <Icon name="thumbsUp" />
                    </button>
                    <button
                      className={`icon-btn danger ${r.rating === "down" ? "active" : ""}`}
                      onClick={() => setRating(t.id, "down")}
                      title="Not helpful"
                      aria-pressed={r.rating === "down"}
                    >
                      <Icon name="thumbsDown" />
                    </button>
                    <button
                      className="icon-btn"
                      onClick={() => {
                        navigator.clipboard?.writeText(t.starter).catch(() => {});
                        trackTopicCopied({
                          topic_id: t.id,
                          topic_category: t.category,
                          target_handle: person.handle,
                        });
                        pushToast({
                          text: "Copied conversation starter.",
                          actionLabel: null,
                          onAction: null,
                          ttl: 2200,
                        });
                      }}
                      title="Copy starter"
                    >
                      <Icon name="copy" />
                    </button>
                  </div>
                </div>
                <p className="topic-pull">{t.starter}</p>
                <p className="topic-why">{t.why}</p>
                <div className="topic-source">
                  <span className="muted-2">Source —</span>
                  {t.url ? (
                    <a href={`https://${t.url.replace(/^https?:\/\//, "")}`} target="_blank" rel="noreferrer">
                      {t.source} <Icon name="extLink" size={11} />
                    </a>
                  ) : (
                    <span>{t.source}</span>
                  )}
                </div>

                {downOpen && (
                  <div className="downvote-panel">
                    <div className="eyebrow">
                      WHAT&apos;S OFF?{" "}
                      <span
                        className="muted-2"
                        style={{ textTransform: "none", letterSpacing: 0, fontSize: 11 }}
                      >
                        (pick one or more)
                      </span>
                    </div>
                    <div className="row">
                      {DOWNVOTE_REASONS.map((reason) => {
                        const sel = (r.reasons || []).includes(reason);
                        return (
                          <span
                            key={reason}
                            className={`chip ${sel ? "selected" : ""}`}
                            onClick={() => toggleReason(t.id, reason)}
                            role="checkbox"
                            aria-checked={sel}
                          >
                            {reason}
                          </span>
                        );
                      })}
                    </div>
                    <textarea
                      className="textarea"
                      placeholder="Optional — anything else? (helps us tune your model)"
                      value={r.note || ""}
                      onChange={(e) => setNote(t.id, e.target.value)}
                      style={{ minHeight: 60 }}
                    />
                    <div className="actions">
                      <button className="btn ghost sm" onClick={() => cancelDownvote(t.id)}>
                        Cancel
                      </button>
                      <button
                        className="btn sm"
                        onClick={() => submitDownvote(t.id)}
                        disabled={(r.reasons || []).length === 0 && !r.note?.trim()}
                      >
                        Submit & remove
                      </button>
                    </div>
                  </div>
                )}
              </article>
            );
          })
        )}
      </div>

      <div
        style={{
          marginTop: 40,
          padding: "24px",
          border: "1px solid var(--line-2)",
          borderRadius: "var(--r-2)",
          background: "var(--surface-2)",
        }}
      >
        <div className="eyebrow" style={{ marginBottom: 10 }}>COLD OUTREACH</div>
        <p className="muted" style={{ fontSize: 13, marginBottom: 16 }}>
          Generate a LinkedIn connection request using the best ice breaker topic.
        </p>
        {outreachStatus === "idle" && (
          <button className="btn secondary sm" onClick={() => void generateOutreach()}>
            Generate message
          </button>
        )}
        {outreachStatus === "loading" && (
          <div className="muted" style={{ fontSize: 13 }}>Generating…</div>
        )}
        {outreachStatus === "error" && (
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <span style={{ fontSize: 13, color: "var(--bad)" }}>Generation failed.</span>
            <button className="btn ghost sm" onClick={() => void generateOutreach()}>Retry</button>
          </div>
        )}
        {outreachStatus === "done" && (
          <div>
            <div
              style={{
                background: "var(--surface)",
                border: "1px solid var(--line-2)",
                borderRadius: "var(--r-1)",
                padding: "14px 16px",
                fontSize: 14,
                lineHeight: 1.6,
                marginBottom: 12,
                whiteSpace: "pre-wrap",
              }}
            >
              {outreachMessage}
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <button
                className="btn sm"
                onClick={() => {
                  navigator.clipboard?.writeText(outreachMessage).catch(() => {});
                  pushToast({ text: "LinkedIn message copied.", ttl: 2200 });
                }}
              >
                <Icon name="copy" /> Copy
              </button>
              <span className="muted-2" style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}>
                {outreachMessage.length} / 300 chars
              </span>
              <button className="btn ghost sm" onClick={() => void generateOutreach()}>
                Regenerate
              </button>
            </div>
          </div>
        )}
      </div>

      <div
        style={{
          marginTop: 48,
          paddingTop: 24,
          borderTop: "1px solid var(--line-2)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div className="muted" style={{ fontSize: 13 }}>
          Generated{" "}
          {relativeTime(state.lastLookup?.ts || new Date().toISOString())} ·{" "}
          {Object.values(state.ratings).filter((rr) => rr.rating === "up").length} marked helpful
        </div>
        <button className="btn secondary sm" onClick={onBackHome}>
          Look up someone else
        </button>
      </div>
    </div>
  );
}
