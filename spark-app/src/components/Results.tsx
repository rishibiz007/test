"use client";
import { useState } from "react";
import { Avatar, Icon } from "./UI";
import { relativeTime } from "@/lib/state";
import type { AppState, Person, RatingState } from "@/lib/types";
import { trackThumbsUp, trackThumbsDownSubmitted, trackTopicCopied } from "@/lib/analytics";

const DOWNVOTE_REASONS = [
  "Too generic",
  "Wrong about them",
  "Awkward / pushy",
  "Already knew this",
  "Tone is off",
  "Not relevant to me",
];

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
      body: JSON.stringify({ topicId, rating, handle: person.handle }),
    }).catch(() => {});
    pushToast({
      text: "Thanks — Spark will surface more like this.",
      actionLabel: "Undo",
      onAction: () => updateRating(topicId, prev),
    });
  };

  const submitDownvote = (topicId: string) => {
    const cur = state.ratings[topicId];
    if (!cur || ((!cur.reasons || cur.reasons.length === 0) && !cur.note?.trim())) return;
    const topic = person.topics.find((t) => t.id === topicId);
    trackThumbsDownSubmitted({
      topic_id: topicId,
      topic_category: topic?.category ?? "",
      target_handle: person.handle,
      reasons: cur.reasons,
      note: cur.note?.trim() ?? "",
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
      }),
    }).catch(() => {});
    pushToast({
      text: "Removed from this list. Spark won't suggest things like this.",
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
              Spark learned from those signals. Try another lookup.
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
