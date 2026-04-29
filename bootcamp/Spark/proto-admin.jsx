/* global React */
// Admin — internal feedback + eval workflow. Side-by-side ideal-output editor.
const { useState: useAdmState, useMemo: useAdmMemo } = React;

const ADMIN_FEEDBACK = [
  {
    id: "fb1",
    when: "2h ago",
    user: "ravi.k@stripe.com",
    target: "Priya Iyer",
    rating: "down",
    reasons: ["Too generic", "Already knew this"],
    note: "Saying 'her SQL post was great' — every PM has seen it. Need something only her closest network would know.",
    starter: "I read your SQL post twice — the framing as leverage, not literacy, is what stuck. Did that come from a specific moment at Stripe?",
    why: "She wrote a long post last week arguing that SQL fluency is now a baseline PM skill — not a nice-to-have.",
  },
  {
    id: "fb2",
    when: "5h ago",
    user: "lila.m@anthropic.com",
    target: "Hugh Park",
    rating: "down",
    reasons: ["Wrong about them"],
    note: "He never worked on Notion AI — he was on the database team. This is fabricated.",
    starter: "Claude's projects feature feels like it has Notion DNA — was that explicit, or did you just end up there?",
    why: "He likely worked on Projects in Claude. The mental model has obvious Notion lineage.",
  },
  {
    id: "fb3",
    when: "yesterday",
    user: "anon",
    target: "Daniel Okonkwo",
    rating: "down",
    reasons: ["Awkward / pushy"],
    note: "Opening with 'spicy entry point' framing comes off as combative, not curious.",
    starter: "Your post about engineers writing PRDs got a lot of pushback in the comments. Did any of it actually change your mind?",
    why: "His post 'Engineers should write the PRD' got 800+ reactions and a heated comment thread.",
  },
  {
    id: "fb4",
    when: "yesterday",
    user: "ravi.k@stripe.com",
    target: "Soraya Nadim",
    rating: "down",
    reasons: ["Tone is off"],
    note: "'Finance bro' is the kind of phrase that lands well with one audience and badly with another. Spark shouldn't put it in someone's mouth.",
    starter: "Ramp's procurement launch felt very different in tone from the original spend cards — less 'finance bro' and more measured. Was that intentional?",
    why: "Ramp announced procurement two weeks ago and the brand voice on the launch page is noticeably different.",
  },
  {
    id: "fb5",
    when: "2 days ago",
    user: "anon",
    target: "Priya Iyer",
    rating: "up",
    reasons: [],
    note: "",
    starter: "Wait, you were Berkeley CS '14? I was '13 — did you ever take Hilfinger's 61B?",
    why: "You both went to Berkeley CS — overlapping years.",
  },
];

const EVAL_SETS = [
  { id: "es1", name: "Generic-output regression", count: 14, lastRun: "this morning", pass: 11, fail: 3 },
  { id: "es2", name: "Hallucinated facts", count: 8, lastRun: "yesterday", pass: 8, fail: 0 },
  { id: "es3", name: "Tone calibration", count: 22, lastRun: "3 days ago", pass: 17, fail: 5 },
];

function AdminPage({ onClose }) {
  const [tab, setTab] = useAdmState("feedback");
  const [selectedFb, setSelectedFb] = useAdmState("fb1");
  const fb = useAdmMemo(() => ADMIN_FEEDBACK.find(f => f.id === selectedFb), [selectedFb]);
  const [ideal, setIdeal] = useAdmState("");
  const [savedAs, setSavedAs] = useAdmState(null);

  React.useEffect(() => { setIdeal(""); setSavedAs(null); }, [selectedFb]);

  const downvotes = ADMIN_FEEDBACK.filter(f => f.rating === "down");

  return (
    <div className="page wide fade-in" data-screen-label="07 admin">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
        <span className="eyebrow">ADMIN · INTERNAL</span>
        <span className="badge" style={{ background: "#1a1a17", borderColor: "#1a1a17", color: "#fbf5e6" }}>SSO · ENG ONLY</span>
      </div>
      <h1 className="h-1 serif" style={{ marginTop: 8, marginBottom: 6 }}>Spark admin</h1>
      <p className="muted" style={{ marginBottom: 22, fontSize: 14 }}>
        Triage user feedback, write ideal outputs, and ship them as evals.
      </p>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--line-2)", marginBottom: 22 }}>
        {[["feedback", "Feedback queue", downvotes.length], ["evals", "Eval sets", EVAL_SETS.length]].map(([k, label, count]) => (
          <button key={k}
            onClick={() => setTab(k)}
            style={{
              border: 0, background: "transparent", cursor: "pointer",
              padding: "8px 14px",
              fontSize: 13, fontWeight: 500,
              color: tab === k ? "var(--ink)" : "var(--ink-3)",
              borderBottom: tab === k ? "2px solid var(--ink)" : "2px solid transparent",
              marginBottom: -1,
            }}>
            {label} <span className="muted-2" style={{ marginLeft: 4, fontFamily: "JetBrains Mono, monospace", fontSize: 11 }}>{count}</span>
          </button>
        ))}
      </div>

      {tab === "feedback" && (
        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 24, alignItems: "start" }}>
          {/* List */}
          <div className="card" style={{ overflow: "hidden" }}>
            <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--line-2)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span className="eyebrow">DOWNVOTES · {downvotes.length}</span>
              <span className="muted-2" style={{ fontSize: 11 }}>last 7 days</span>
            </div>
            {downvotes.map(f => (
              <button key={f.id}
                onClick={() => setSelectedFb(f.id)}
                style={{
                  display: "block", width: "100%", textAlign: "left",
                  padding: "12px 14px",
                  borderBottom: "1px solid var(--line-2)",
                  border: 0,
                  background: selectedFb === f.id ? "var(--bg-2)" : "var(--bg)",
                  cursor: "pointer",
                  borderLeft: selectedFb === f.id ? "2px solid var(--ink)" : "2px solid transparent",
                }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                  <span style={{ fontSize: 13, color: "var(--ink)", fontWeight: 500 }}>{f.target}</span>
                  <span className="muted-2" style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}>{f.when}</span>
                </div>
                <div className="muted" style={{ fontSize: 12, marginBottom: 6, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {f.starter}
                </div>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  {f.reasons.map(r => <span key={r} className="chip" style={{ height: 18, fontSize: 10, padding: "0 7px", cursor: "default" }}>{r}</span>)}
                </div>
              </button>
            ))}
          </div>

          {/* Side-by-side editor */}
          {fb && (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                <span className="tag">FROM</span>
                <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 12, color: "var(--ink-2)" }}>{fb.user}</span>
                <span className="muted-2" style={{ fontSize: 11 }}>·</span>
                <span className="muted" style={{ fontSize: 12 }}>about <strong style={{ color: "var(--ink-2)" }}>{fb.target}</strong></span>
              </div>

              {fb.note && (
                <div style={{ padding: "12px 14px", background: "var(--bad-soft)", border: "1px solid #f0c8c4", borderRadius: "var(--r-1)", marginBottom: 18, fontSize: 13, color: "#8a3933" }}>
                  <div className="eyebrow" style={{ color: "#a14a44", marginBottom: 4 }}>USER NOTE</div>
                  "{fb.note}"
                </div>
              )}

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {/* Original */}
                <div>
                  <div className="eyebrow" style={{ marginBottom: 8 }}>WHAT SPARK SAID</div>
                  <div className="card" style={{ padding: 16, opacity: 0.9 }}>
                    <p className="topic-pull" style={{ fontSize: 16, marginBottom: 10 }}>{fb.starter}</p>
                    <div className="muted" style={{ fontSize: 12 }}>{fb.why}</div>
                  </div>
                </div>

                {/* Ideal */}
                <div>
                  <div className="eyebrow" style={{ marginBottom: 8, color: "var(--accent)" }}>YOUR IDEAL OUTPUT</div>
                  <div className="card" style={{ padding: 16, borderColor: "var(--accent-line)", background: "#fffefa" }}>
                    <textarea
                      className="textarea"
                      placeholder="Write what Spark should have said instead…"
                      value={ideal}
                      onChange={(e) => setIdeal(e.target.value)}
                      style={{ minHeight: 120, fontFamily: "Source Serif 4, Georgia, serif", fontSize: 16, fontStyle: "italic", border: 0, padding: 0, background: "transparent" }}
                    />
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 22, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="muted" style={{ fontSize: 12 }}>
                  Save as eval row · adds to <strong style={{ color: "var(--ink-2)" }}>Generic-output regression</strong>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn ghost sm">Skip</button>
                  <button
                    className="btn sm"
                    disabled={!ideal.trim()}
                    onClick={() => setSavedAs(`Saved to "Generic-output regression" · row #${15 + Math.floor(Math.random() * 5)}`)}
                  >
                    <window.UI.Icon name="check" /> Save as eval
                  </button>
                </div>
              </div>
              {savedAs && (
                <div style={{ marginTop: 14, padding: "10px 14px", background: "var(--good-soft)", border: "1px solid #c1e0cb", color: "#226639", fontSize: 13, borderRadius: "var(--r-1)" }}>
                  ✓ {savedAs}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {tab === "evals" && (
        <div className="card">
          {EVAL_SETS.map((s, i) => (
            <div key={s.id} style={{
              display: "grid", gridTemplateColumns: "1fr auto auto auto auto",
              gap: 18, alignItems: "center",
              padding: "14px 18px",
              borderBottom: i < EVAL_SETS.length - 1 ? "1px solid var(--line-2)" : 0,
            }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 500 }}>{s.name}</div>
                <div className="muted" style={{ fontSize: 12 }}>{s.count} rows · last run {s.lastRun}</div>
              </div>
              <span className="badge success">{s.pass} pass</span>
              {s.fail > 0
                ? <span className="badge" style={{ background: "var(--bad-soft)", borderColor: "#f0c8c4", color: "#8a3933" }}>{s.fail} fail</span>
                : <span style={{ width: 0 }} />}
              <button className="btn secondary sm">Open</button>
              <button className="btn sm">Run</button>
            </div>
          ))}
          <div style={{ padding: "12px 18px", display: "flex", justifyContent: "flex-end" }}>
            <button className="btn ghost sm"><window.UI.Icon name="plus" /> New eval set</button>
          </div>
        </div>
      )}
    </div>
  );
}

window.AdminPage = AdminPage;
