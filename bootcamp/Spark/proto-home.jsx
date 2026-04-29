/* global React */
// Home (Lookup) + History + loading screen + Profile.
const { useState: useHomeState, useMemo: useHomeMemo, useEffect: useHomeEffect, useRef: useHomeRef } = React;

function Home({ store, onLookup, onOpenHistory, onOpenLookup, toast }) {
  const [url, setUrl] = useHomeState("");
  const [error, setError] = useHomeState("");
  const inputRef = useHomeRef(null);

  // ⌘K to focus
  useHomeEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current && inputRef.current.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const submit = () => {
    const cleaned = url.trim().replace(/^https?:\/\//, "").replace(/\/$/, "");
    if (!cleaned) return;
    if (!cleaned.includes("linkedin.com/in/")) {
      setError("That doesn't look like a LinkedIn profile URL.");
      return;
    }
    if (!window.SparkData.PEOPLE[cleaned]) {
      setError(`We don't have demo data for "${cleaned}". Try one of the suggested profiles below.`);
      return;
    }
    setError("");
    setUrl("");
    onLookup(cleaned);
  };

  const recent = store.history.slice(0, 4);

  const suggestions = Object.values(window.SparkData.PEOPLE).map(p => p.handle);

  return (
    <div className="page narrow fade-in" data-screen-label="02 home">
      <div style={{ marginBottom: 32 }}>
        <span className="eyebrow">NEW LOOKUP</span>
      </div>
      <h1 className="h-display serif" style={{ marginBottom: 14 }}>
        Who are you meeting?
      </h1>
      <p className="muted" style={{ marginBottom: 28, fontSize: 15, maxWidth: 520 }}>
        Paste a LinkedIn URL. Spark reads their profile, recent posts, and any podcast or blog appearances —
        then surfaces what you actually have in common.
      </p>

      <div style={{ position: "relative" }}>
        <div className="search-wrap" style={{ height: 44, padding: "0 8px 0 14px" }}>
          <window.UI.Icon name="search" size={15} />
          <input
            ref={inputRef}
            placeholder="linkedin.com/in/…"
            value={url}
            onChange={(e) => { setUrl(e.target.value); setError(""); }}
            onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
            className="mono"
            style={{ fontSize: 13 }}
          />
          <span className="kbd" style={{ marginRight: 8 }}>⌘ K</span>
          <button className="btn" onClick={submit} disabled={!url.trim()}>
            Look up <window.UI.Icon name="arrow" />
          </button>
        </div>
        {error && (
          <div style={{
            marginTop: 10, fontSize: 12, color: "var(--bad)",
            background: "var(--bad-soft)", border: "1px solid #f0c8c4",
            padding: "8px 12px", borderRadius: "var(--r-1)",
          }}>{error}</div>
        )}
        <div className="muted-2" style={{ fontSize: 11, marginTop: 10, fontFamily: "JetBrains Mono, monospace" }}>
          public info only · personalized using your profile · cached 24h
        </div>
      </div>

      {/* Suggested profiles for demo */}
      <div style={{ marginTop: 28 }}>
        <div className="eyebrow" style={{ marginBottom: 10 }}>TRY ONE OF THESE</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {suggestions.map(h => {
            const p = window.SparkData.PEOPLE[h];
            return (
              <button key={h} className="chip" onClick={() => { setUrl(h); setError(""); setTimeout(() => onLookup(h), 60); }}>
                {p.name} · {p.company}
              </button>
            );
          })}
        </div>
      </div>

      {/* Recents */}
      <div style={{ marginTop: 48 }}>
        <div className="sec-head">
          <span className="eyebrow">RECENT</span>
          {store.history.length > 4 && (
            <a href="#" className="link" onClick={(e) => { e.preventDefault(); onOpenHistory(); }}>See all →</a>
          )}
        </div>
        {recent.length === 0 ? (
          <div className="empty">
            <div className="em-title">No lookups yet.</div>
            <div style={{ fontSize: 13 }}>Try one of the suggested profiles above to get started.</div>
          </div>
        ) : (
          <div className="card stagger">
            {recent.map((it, i) => {
              const p = window.SparkData.PEOPLE[it.handle];
              if (!p) return null;
              return (
                <a key={i} href="#" className="history-row" onClick={(e) => { e.preventDefault(); onOpenLookup(it.handle); }}>
                  <window.UI.Avatar initials={p.initials} size="md" />
                  <div style={{ minWidth: 0 }}>
                    <div className="person-name">{p.name}</div>
                    <div className="person-co">{p.company}</div>
                  </div>
                  <span className="muted-2" style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}>
                    {window.SparkData.relativeTime(it.when)}
                  </span>
                  <span className="badge success" title="Cached snapshot — no re-scrape">cached</span>
                </a>
              );
            })}
            <a href="#" className="history-row" onClick={(e) => { e.preventDefault(); onOpenHistory(); }}
              style={{ color: "var(--ink-3)", borderTop: "1px solid var(--line-2)" }}>
              <span style={{ width: 36, textAlign: "center", color: "var(--ink-4)" }}>↗</span>
              <div className="person-name" style={{ color: "var(--ink-2)", fontSize: 13 }}>See all lookups</div>
              <span/>
              <span/>
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

function LoadingScreen({ handle, onDone }) {
  const person = window.SparkData.PEOPLE[handle];
  const steps = [
    "Reading their profile",
    `Checking recent posts (${person?.topics.filter(t => t.category === "RECENT ACTIVITY").length || 2} found)`,
    "Scanning podcast & blog mentions",
    "Finding overlap with your profile",
  ];
  const [active, setActive] = useHomeState(0);
  useHomeEffect(() => {
    const timers = [];
    [620, 1180, 1820, 2480].forEach((ms, i) => {
      timers.push(setTimeout(() => setActive(i + 1), ms));
    });
    timers.push(setTimeout(onDone, 3000));
    return () => timers.forEach(clearTimeout);
  }, []);
  return (
    <div className="loading fade-in" data-screen-label="03 loading">
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 28 }}>
        <window.UI.Avatar initials={person?.initials || "??"} size="lg" />
        <div>
          <div className="serif" style={{ fontSize: 22, lineHeight: 1.1 }}>{person?.name}</div>
          <div className="muted" style={{ fontSize: 13 }}>{person?.role} · {person?.company}</div>
        </div>
      </div>
      <div>
        {steps.map((s, i) => (
          <div key={s} className={`progress-line ${i < active ? "done" : i === active ? "active" : "pending"}`}>
            <span className="progress-dot" />
            <span>{i < active ? "✓ " : i === active ? "→ " : "  "}{s}{i === active && "…"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HistoryPage({ store, onOpenLookup, onBackHome }) {
  const [q, setQ] = useHomeState("");
  const filtered = useHomeMemo(() => {
    const items = store.history.map(it => ({ ...it, person: window.SparkData.PEOPLE[it.handle] })).filter(x => x.person);
    if (!q.trim()) return items;
    const t = q.toLowerCase();
    return items.filter(x =>
      x.person.name.toLowerCase().includes(t) ||
      x.person.company.toLowerCase().includes(t) ||
      x.handle.toLowerCase().includes(t)
    );
  }, [q, store.history]);

  return (
    <div className="page narrow fade-in" data-screen-label="04 history">
      <div style={{ marginBottom: 22 }}>
        <span className="eyebrow">HISTORY</span>
      </div>
      <h1 className="h-1 serif" style={{ marginBottom: 22 }}>Your lookups</h1>
      <div className="search-wrap" style={{ marginBottom: 22 }}>
        <window.UI.Icon name="search" size={15} />
        <input placeholder="Search by name, company, or URL…" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
      </div>
      {filtered.length === 0 ? (
        <div className="empty">
          <div className="em-title">{q ? "No matches" : "Nothing here yet."}</div>
          <div style={{ fontSize: 13 }}>{q ? "Try a different search term." : "Look up someone to start filling this list."}</div>
          {!q && <button className="btn secondary sm" style={{ marginTop: 14 }} onClick={onBackHome}>Look someone up →</button>}
        </div>
      ) : (
        <div className="card">
          {filtered.map((it, i) => (
            <a key={i} href="#" className="history-row" onClick={(e) => { e.preventDefault(); onOpenLookup(it.handle); }}>
              <window.UI.Avatar initials={it.person.initials} size="md" />
              <div style={{ minWidth: 0 }}>
                <div className="person-name">{it.person.name}</div>
                <div className="person-co">{it.person.company} · {it.person.role}</div>
              </div>
              <span className="muted-2" style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}>
                {window.SparkData.relativeTime(it.when)}
              </span>
              <span className="badge success">cached</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function ProfilePage({ store }) {
  const u = store.user;
  return (
    <div className="page narrow fade-in" data-screen-label="05 profile">
      <span className="eyebrow">YOUR PROFILE</span>
      <h1 className="h-1 serif" style={{ marginTop: 8, marginBottom: 22 }}>How Spark sees you</h1>

      <div className="card" style={{
        padding: "12px 16px", display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "var(--bg-2)", marginBottom: 22, fontSize: 13,
      }}>
        <span className="muted">Last refreshed from LinkedIn — <strong style={{ color: "var(--ink-2)" }}>{u.refreshedAt}</strong>.</span>
        <button className="btn secondary sm" disabled title="Already refreshed today — try tomorrow.">Refresh</button>
      </div>

      <div className="card" style={{ padding: 22 }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 18 }}>
          <window.UI.Avatar initials={u.initials} size="xl" tone="amber" />
          <div>
            <div style={{ fontSize: 18, fontWeight: 500 }}>{u.name}</div>
            <div className="muted" style={{ fontSize: 13 }}>{u.role}</div>
            <div className="mono" style={{ fontSize: 11, color: "var(--ink-4)", marginTop: 4 }}>{u.linkedin}</div>
          </div>
        </div>
        <hr className="divider" style={{ margin: "8px 0" }} />
        <div className="profile-row"><span className="label">Email</span><span>{u.email}</span><span className="muted-2" style={{ fontSize: 11 }}>edit</span></div>
        <div className="profile-row"><span className="label">Education</span><span>{u.education}</span><span className="muted-2" style={{ fontSize: 11 }}>edit</span></div>
        <div className="profile-row"><span className="label">Recent</span><span style={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.recentPosts}</span><span className="muted-2" style={{ fontSize: 11 }}>edit</span></div>
        <div className="profile-row"><span className="label">Podcast</span><span>{u.podcasts}</span><span className="muted-2" style={{ fontSize: 11 }}>edit</span></div>

        <div style={{ marginTop: 22 }}>
          <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>Looking for</label>
          <textarea className="textarea" defaultValue={u.lookingFor} />
        </div>
        <div style={{ marginTop: 16 }}>
          <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>Love talking about</label>
          <textarea className="textarea" defaultValue={u.talksAbout} />
        </div>
      </div>
    </div>
  );
}

window.Home = Home;
window.LoadingScreen = LoadingScreen;
window.HistoryPage = HistoryPage;
window.ProfilePage = ProfilePage;
