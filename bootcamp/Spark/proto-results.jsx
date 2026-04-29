/* global React */
// Editorial results page with persisted ratings, downvote panel, undo, fade.
const { useState: useResState, useMemo: useResMemo, useEffect: useResEffect } = React;

const DOWNVOTE_REASONS = [
  "Too generic",
  "Wrong about them",
  "Awkward / pushy",
  "Already knew this",
  "Tone is off",
  "Not relevant to me",
];

function Results({ store, handle, onBackHome, pushToast }) {
  const person = window.SparkData.PEOPLE[handle];
  const [openPanel, setOpenPanel] = useResState(null); // topicId
  const [removing, setRemoving] = useResState({}); // topicId -> true (animating out)

  if (!person) {
    return (
      <div className="page narrow"><div className="empty">
        <div className="em-title">Profile not found</div>
        <button className="btn secondary sm" style={{marginTop:12}} onClick={onBackHome}>← Back home</button>
      </div></div>
    );
  }

  const visibleTopics = person.topics.filter(t => {
    const r = store.ratings[t.id];
    if (r?.rating === "down" && r.reasons && r.reasons.length > 0) return false; // submitted
    return !removing[t.id];
  });

  const setRating = (topicId, rating) => {
    const prev = store.ratings[topicId] || {};
    if (prev.rating === rating) {
      // toggle off
      store.update({ ratings: { ...store.ratings, [topicId]: { ...prev, rating: null } } });
      return;
    }
    if (rating === "down") {
      // open panel; don't commit yet
      setOpenPanel(topicId);
      store.update({ ratings: { ...store.ratings, [topicId]: { ...prev, rating: "down", reasons: prev.reasons || [], note: prev.note || "" } } });
      return;
    }
    store.update({ ratings: { ...store.ratings, [topicId]: { rating, reasons: [], note: "" } } });
    pushToast({
      text: rating === "up" ? "Thanks — Spark will surface more like this." : "Marked.",
      actionLabel: "Undo",
      onAction: () => {
        store.update({ ratings: { ...store.ratings, [topicId]: { ...prev } } });
      },
    });
  };

  const submitDownvote = (topicId) => {
    const cur = store.ratings[topicId];
    if (!cur || !cur.reasons || cur.reasons.length === 0) {
      // require a reason
      return;
    }
    setOpenPanel(null);
    setRemoving(prev => ({ ...prev, [topicId]: true }));
    pushToast({
      text: "Removed from this list. Spark won't suggest things like this.",
      actionLabel: "Undo",
      onAction: () => {
        const restore = { ...store.ratings };
        delete restore[topicId];
        store.update({ ratings: restore });
        setRemoving(prev => { const n = { ...prev }; delete n[topicId]; return n; });
      },
    });
  };

  const cancelDownvote = (topicId) => {
    const restore = { ...store.ratings };
    delete restore[topicId];
    store.update({ ratings: restore });
    setOpenPanel(null);
  };

  const toggleReason = (topicId, reason) => {
    const cur = store.ratings[topicId] || { rating: "down", reasons: [], note: "" };
    const reasons = cur.reasons.includes(reason)
      ? cur.reasons.filter(r => r !== reason)
      : [...cur.reasons, reason];
    store.update({ ratings: { ...store.ratings, [topicId]: { ...cur, reasons } } });
  };

  const setNote = (topicId, note) => {
    const cur = store.ratings[topicId] || { rating: "down", reasons: [], note: "" };
    store.update({ ratings: { ...store.ratings, [topicId]: { ...cur, note } } });
  };

  return (
    <div className="page fade-in" data-screen-label="06 results">
      <button className="btn ghost sm" onClick={onBackHome} style={{ marginLeft: -10, marginBottom: 14 }}>
        ← New lookup
      </button>

      <div className="editorial-meta">
        <window.UI.Avatar initials={person.initials} size="xl" />
        <div>
          <div className="eyebrow" style={{ marginBottom: 6 }}>SPARK BRIEF · {person.topics.length} TOPICS</div>
          <h1 className="h-display serif" style={{ marginBottom: 6 }}>{person.name}</h1>
          <div className="muted" style={{ fontSize: 14 }}>{person.bio}</div>
        </div>
        <span className="badge success" title="Cached for 24 hours">CACHED</span>
      </div>

      <div className="stagger" style={{ position: "relative" }}>
        {visibleTopics.length === 0 ? (
          <div className="empty">
            <div className="em-title">All topics dismissed</div>
            <div style={{ fontSize: 13 }}>Spark learned from those signals. Try another lookup.</div>
            <button className="btn secondary sm" style={{ marginTop: 14 }} onClick={onBackHome}>Look someone else up →</button>
          </div>
        ) : visibleTopics.map((t, i) => {
          const r = store.ratings[t.id] || {};
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
                {t.usedYou && <span className="badge" title="Personalized using your profile">YOUR PROFILE</span>}
                <div className="topic-actions">
                  <button
                    className={`icon-btn ${liked ? "active" : ""}`}
                    onClick={() => setRating(t.id, "up")}
                    title="Helpful"
                    aria-pressed={liked}
                  >
                    <window.UI.Icon name="thumbsUp" />
                  </button>
                  <button
                    className={`icon-btn danger ${r.rating === "down" ? "active" : ""}`}
                    onClick={() => setRating(t.id, "down")}
                    title="Not helpful"
                    aria-pressed={r.rating === "down"}
                  >
                    <window.UI.Icon name="thumbsDown" />
                  </button>
                  <button
                    className="icon-btn"
                    onClick={() => {
                      navigator.clipboard?.writeText(t.starter).catch(() => {});
                      pushToast({ text: "Copied conversation starter.", actionLabel: null, onAction: null, ttl: 2200 });
                    }}
                    title="Copy starter"
                  >
                    <window.UI.Icon name="copy" />
                  </button>
                </div>
              </div>
              <p className="topic-pull">{t.starter}</p>
              <p className="topic-why">{t.why}</p>
              <div className="topic-source">
                <span className="muted-2">Source —</span>
                {t.url ? (
                  <a href={`https://${t.url}`} target="_blank" rel="noreferrer">
                    {t.source} <window.UI.Icon name="extLink" size={11} />
                  </a>
                ) : (
                  <span>{t.source}</span>
                )}
              </div>

              {downOpen && (
                <div className="downvote-panel">
                  <div className="eyebrow">WHAT'S OFF? <span className="muted-2" style={{ textTransform: "none", letterSpacing: 0, fontSize: 11 }}>(pick one or more)</span></div>
                  <div className="row">
                    {DOWNVOTE_REASONS.map(reason => {
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
                    <button className="btn ghost sm" onClick={() => cancelDownvote(t.id)}>Cancel</button>
                    <button className="btn sm" onClick={() => submitDownvote(t.id)} disabled={(r.reasons || []).length === 0}>
                      Submit & remove
                    </button>
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>

      <div style={{ marginTop: 48, paddingTop: 24, borderTop: "1px solid var(--line-2)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div className="muted" style={{ fontSize: 13 }}>
          Generated {window.SparkData.relativeTime(store.lastLookup?.ts || new Date().toISOString())} ·
          {" "}{Object.values(store.ratings).filter(r => r.rating === "up").length} marked helpful
        </div>
        <button className="btn secondary sm" onClick={onBackHome}>Look up someone else</button>
      </div>
    </div>
  );
}

window.Results = Results;
