/* global React, SPARK_DATA */
// Admin dashboard — 2 variants on the eval workflow + cache inspector
const { useState: useAdminState } = React;

function Admin({ variant, density }) {
  const compact = density === "compact";
  const [tab, setTab] = useAdminState("feedback");

  return (
    <div className={`page ${compact ? "compact" : ""}`} style={{maxWidth: "100%", paddingLeft: 20, paddingRight: 20}}>
      <div className="page-note">
        wireframe · admin · variant: <strong style={{color:"var(--ink-2)"}}>
          {variant === "builder" ? "Eval set builder (select rows → write ideal → export)" : "Side-by-side: original output vs ideal output editor"}
        </strong>
      </div>

      <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom: 18}}>
        <div style={{display:"flex", alignItems:"center", gap:12}}>
          <div style={{
            fontFamily:"JetBrains Mono, monospace", fontSize:11, color:"var(--ink-3)",
            border:"1px solid var(--line)", padding:"3px 8px", borderRadius:3,
          }}>internal · password-gated</div>
          <h1 className="h1 serif" style={{fontSize: 22, margin:0}}>Spark admin</h1>
        </div>
        <AdminTabs tab={tab} setTab={setTab} />
      </div>

      {tab === "feedback" && (variant === "builder" ? <EvalBuilder compact={compact}/> : <EvalSideBySide compact={compact}/>)}
      {tab === "cache" && <CacheInspector compact={compact} />}
    </div>
  );
}

function AdminTabs({ tab, setTab }) {
  const tabs = [
    ["feedback", "Feedback + eval"],
    ["cache", "Cache inspector"],
  ];
  return (
    <div style={{display:"flex", gap:4}}>
      {tabs.map(([id, label]) => (
        <button key={id} onClick={() => setTab(id)} className={`variant-tab ${tab === id ? "active" : ""}`}>
          {label}
        </button>
      ))}
    </div>
  );
}

/* ---------- Variant A: Eval Builder ---------- */
function EvalBuilder({ compact }) {
  const F = window.SPARK_DATA.FEEDBACK;
  const E = window.SPARK_DATA.EVAL_SETS;
  const [selected, setSelected] = useAdminState(["f1", "f3", "f5"]);
  const [activeSet, setActiveSet] = useAdminState("e1");

  const toggle = (id) => {
    setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id]);
  };

  return (
    <div style={{display:"grid", gridTemplateColumns: "1fr 280px", gap: 18}}>
      {/* Feedback feed table */}
      <div className="card" style={{overflow:"hidden"}}>
        <FilterBar />
        <table className="feedback-table">
          <thead>
            <tr>
              <th style={{width:30}}></th>
              <th>Topic</th>
              <th style={{width:120}}>Category</th>
              <th style={{width:60}}>Rating</th>
              <th style={{width:160}}>Reason tags</th>
              <th style={{width:200}}>Free-text</th>
              <th style={{width:140}}>Person</th>
              <th style={{width:80}}>Date</th>
            </tr>
          </thead>
          <tbody>
            {F.map(f => (
              <FeedbackRow key={f.id} f={f} checked={selected.includes(f.id)} onToggle={() => toggle(f.id)} />
            ))}
          </tbody>
        </table>
        <div style={{
          padding:"10px 14px", background:"var(--bg-2)", borderTop:"1px solid var(--line-2)",
          display:"flex", justifyContent:"space-between", alignItems:"center", fontSize:12, color:"var(--ink-2)",
        }}>
          <span>{selected.length} selected · 7 of 412 rows shown</span>
          <div style={{display:"flex", gap:6}}>
            <button className="btn subtle sm">Clear</button>
            <button className="btn sm">+ Add to {E.find(e => e.id === activeSet)?.name}</button>
          </div>
        </div>
      </div>

      {/* Eval set sidebar */}
      <div style={{display:"flex", flexDirection:"column", gap:14}}>
        <div className="card" style={{padding:14}}>
          <div className="tag" style={{marginBottom:10}}>EVAL SETS</div>
          <div style={{display:"flex", flexDirection:"column", gap:2}}>
            {E.map(s => (
              <button key={s.id} onClick={() => setActiveSet(s.id)} style={{
                textAlign:"left",
                padding:"8px 10px",
                border:"none",
                background: activeSet === s.id ? "var(--bg-2)" : "transparent",
                borderRadius: 4,
                cursor:"pointer",
                display:"flex", justifyContent:"space-between", alignItems:"center",
                fontSize: 13,
                color: "var(--ink)",
              }}>
                <span style={{fontFamily:"JetBrains Mono, monospace", fontSize:12}}>{s.name}</span>
                <span style={{color:"var(--ink-3)", fontSize:11}}>{s.count}</span>
              </button>
            ))}
          </div>
          <button className="btn subtle sm" style={{marginTop:10, width:"100%", justifyContent:"center"}}>+ New eval set</button>
        </div>

        <div className="card" style={{padding:14}}>
          <div className="tag" style={{marginBottom:10}}>ACTIVE: v1-baseline-failures</div>
          <div style={{fontSize:12, color:"var(--ink-2)", lineHeight:1.5, marginBottom:10}}>
            12 entries · created Apr 18, 2026 · 4 still need ideal output
          </div>
          <div style={{display:"flex", flexDirection:"column", gap:6}}>
            <button className="btn sm" style={{justifyContent:"center"}}>↓ Export as JSON</button>
            <button className="btn subtle sm" style={{justifyContent:"center"}}>Edit ideal outputs →</button>
          </div>
        </div>

        {/* Inline ideal-output editor for the most recently added row */}
        <div className="card" style={{padding:14, background:"var(--bg-2)"}}>
          <div className="tag" style={{marginBottom:8}}>WRITE IDEAL OUTPUT</div>
          <div className="serif" style={{fontSize:13, color:"var(--ink-2)", marginBottom:8, lineHeight:1.4, fontStyle:"italic"}}>
            "Loved your hot take on agile being a vibe."
          </div>
          <textarea className="textarea" style={{fontSize:12, fontFamily:"Source Serif 4, Georgia, serif"}}
            defaultValue="(write what we wish the model had said for this person — be specific, source-anchored, in the user's voice)"
          />
          <div style={{display:"flex", justifyContent:"flex-end", marginTop:8}}>
            <button className="btn sm">Save</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function FilterBar() {
  return (
    <div style={{
      padding:"10px 14px", borderBottom:"1px solid var(--line-2)",
      display:"flex", gap:8, alignItems:"center", flexWrap:"wrap",
    }}>
      <span className="tag" style={{marginRight:6}}>FILTERS</span>
      {["Rating: all", "Category: all", "Reason: all", "Date: last 30d"].map((f) => (
        <span key={f} style={{
          fontSize:12, padding:"3px 9px", border:"1px solid var(--line)", borderRadius:99,
          color:"var(--ink-2)", background:"var(--bg)", cursor:"pointer",
        }}>{f} ▾</span>
      ))}
      <div style={{flex:1}}/>
      <span style={{fontSize:12, color:"var(--ink-3)"}}>Sort:</span>
      <span style={{fontSize:12, padding:"3px 9px", border:"1px solid var(--line)", borderRadius:3, color:"var(--ink-2)"}}>Date ↓</span>
    </div>
  );
}

function FeedbackRow({ f, checked, onToggle }) {
  return (
    <tr style={{background: checked ? "var(--accent-soft)" : "transparent"}}>
      <td>
        <input type="checkbox" checked={checked} onChange={onToggle} />
      </td>
      <td>
        <span className="serif" style={{fontSize:13, color:"var(--ink)"}}>{f.topic}</span>
      </td>
      <td><span className="tag" style={{fontSize:9}}>{f.category}</span></td>
      <td>
        <span style={{
          color: f.rating === "up" ? "var(--good)" : "var(--bad)",
          fontSize: 14,
        }}>{f.rating === "up" ? "👍" : "👎"}</span>
      </td>
      <td>
        <div style={{display:"flex", flexWrap:"wrap", gap:4}}>
          {f.tags.length === 0 && <span style={{color:"var(--ink-4)", fontSize:11}}>—</span>}
          {f.tags.map(t => (
            <span key={t} style={{
              fontSize:10, padding:"2px 7px", border:"1px solid var(--line)", borderRadius:99,
              color:"var(--ink-2)",
            }}>{t}</span>
          ))}
        </div>
      </td>
      <td style={{fontSize:12, color:"var(--ink-2)", fontStyle:"italic"}}>{f.text || <span style={{color:"var(--ink-4)", fontStyle:"normal"}}>—</span>}</td>
      <td style={{fontSize:12, color:"var(--ink-2)"}}>{f.person}</td>
      <td className="mono" style={{fontSize:11, color:"var(--ink-3)"}}>{f.date}</td>
    </tr>
  );
}

/* ---------- Variant B: Side-by-side editor ---------- */
function EvalSideBySide({ compact }) {
  const F = window.SPARK_DATA.FEEDBACK.filter(f => f.rating === "down");

  return (
    <div style={{display:"grid", gridTemplateColumns: "260px 1fr", gap: 18, height: "calc(100vh - 220px)"}}>
      {/* Left: queue of failures */}
      <div className="card" style={{overflow:"auto", display:"flex", flexDirection:"column"}}>
        <div style={{padding: "12px 14px", borderBottom: "1px solid var(--line-2)", background:"var(--bg-2)"}}>
          <div className="tag">QUEUE · v1-baseline-failures</div>
          <div style={{fontSize:11, color:"var(--ink-3)", marginTop:4}}>{F.length} entries · 2 with ideal output</div>
        </div>
        {F.map((f, i) => (
          <div key={f.id} style={{
            padding:"12px 14px",
            borderBottom: "1px solid var(--line-2)",
            background: i === 0 ? "var(--bg-2)" : "transparent",
            cursor:"pointer",
            borderLeft: i === 0 ? "3px solid var(--accent)" : "3px solid transparent",
          }}>
            <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:6}}>
              <span className="tag" style={{fontSize:9}}>{f.category}</span>
              <span style={{
                fontSize:10, fontFamily:"JetBrains Mono, monospace",
                color: i < 2 ? "var(--good)" : "var(--ink-4)",
              }}>{i < 2 ? "✓ ideal" : "○ todo"}</span>
            </div>
            <div className="serif" style={{fontSize:13, lineHeight:1.4, color:"var(--ink)"}}>
              {f.topic}
            </div>
            <div style={{fontSize:11, color:"var(--ink-3)", marginTop:6}}>{f.person}</div>
          </div>
        ))}
      </div>

      {/* Right: side-by-side editor */}
      <div className="card" style={{display:"flex", flexDirection:"column", overflow:"hidden"}}>
        {/* Header */}
        <div style={{padding:"14px 18px", borderBottom:"1px solid var(--line-2)", display:"flex", alignItems:"center", justifyContent:"space-between"}}>
          <div>
            <div style={{fontSize:13, color:"var(--ink-2)"}}>
              <span className="mono" style={{fontSize:11, color:"var(--ink-3)"}}>entry 1 / 4</span>
              <span style={{margin:"0 10px", color:"var(--ink-4)"}}>·</span>
              <span>Priya Iyer · Stripe</span>
              <span style={{margin:"0 10px", color:"var(--ink-4)"}}>·</span>
              <span style={{color:"var(--bad)"}}>👎 too generic, wrong tone</span>
            </div>
          </div>
          <div style={{display:"flex", gap:6}}>
            <button className="btn subtle sm">← Prev</button>
            <button className="btn subtle sm">Next →</button>
            <button className="btn sm">↓ Export set</button>
          </div>
        </div>

        {/* Two-pane editor */}
        <div style={{flex:1, display:"grid", gridTemplateColumns:"1fr 1fr", overflow:"hidden"}}>
          {/* Original output */}
          <div style={{padding:"18px 22px", borderRight:"1px solid var(--line-2)", overflow:"auto"}}>
            <div className="tag" style={{marginBottom:10}}>ORIGINAL OUTPUT · what the model said</div>
            <p className="serif" style={{fontSize:16, lineHeight:1.5, color:"var(--ink)", margin:"0 0 18px"}}>
              Loved your hot take on agile being a vibe.
            </p>
            <hr className="divider" style={{margin:"14px 0"}}/>

            <div className="tag" style={{marginBottom:10}}>USER FEEDBACK</div>
            <div style={{display:"flex", flexWrap:"wrap", gap:6, marginBottom:10}}>
              {["Too generic", "Wrong tone"].map(t => (
                <span key={t} style={{
                  fontSize:11, padding:"3px 8px", borderRadius:99,
                  border:"1px solid var(--line)", color:"var(--ink-2)",
                }}>{t}</span>
              ))}
            </div>
            <p style={{fontSize:13, color:"var(--ink-2)", fontStyle:"italic", lineHeight:1.5}}>
              "Sounds AI-generated. 'Hot take' is a tell."
            </p>

            <hr className="divider" style={{margin:"18px 0 14px"}}/>

            <div className="tag" style={{marginBottom:10}}>INPUT SNAPSHOT (truncated)</div>
            <pre className="mono" style={{
              fontSize:11, color:"var(--ink-2)", margin:0,
              background:"var(--bg-2)", padding:12, borderRadius:4,
              whiteSpace:"pre-wrap", lineHeight:1.5,
            }}>
{`{
  "user": { "linkedin": "linkedin.com/in/mayapatel", … },
  "target": { "linkedin": "linkedin.com/in/priyaiyer", … },
  "target_post": "Lately I've been thinking about why
    agile rituals never seem to ship value…"
}`}
            </pre>
          </div>

          {/* Ideal output (editable) */}
          <div style={{padding:"18px 22px", overflow:"auto", background:"var(--bg-2)"}}>
            <div className="tag" style={{marginBottom:10}}>IDEAL OUTPUT · what we wish it had said</div>
            <textarea
              className="textarea"
              style={{
                background:"var(--bg)",
                fontFamily:"Source Serif 4, Georgia, serif",
                fontSize:15,
                lineHeight:1.5,
                minHeight: 140,
                marginBottom: 14,
              }}
              defaultValue={`I read your agile post twice — what stuck wasn't the hot-take framing, it was the bit about how rituals fossilize when teams stop questioning why they exist. Did that come from a specific moment at Stripe, or is it more of a slow accumulation?`}
            />

            <div className="tag" style={{marginBottom:10}}>NOTES (optional)</div>
            <textarea
              className="textarea"
              style={{fontSize:13, minHeight: 70, background:"var(--bg)"}}
              defaultValue="Be specific about the source. Avoid 'hot take', 'vibe', 'crushed it'. Always end with an open question."
            />

            <div style={{display:"flex", gap:6, marginTop:12, justifyContent:"flex-end"}}>
              <button className="btn subtle sm">Skip</button>
              <button className="btn sm">Save & next →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Cache inspector ---------- */
function CacheInspector({ compact }) {
  const rows = [
    { url: "linkedin.com/in/priyaiyer", date: "2026-04-28", today: true, sizeKB: 24 },
    { url: "linkedin.com/in/priyaiyer", date: "2026-04-26", today: false, sizeKB: 22 },
    { url: "linkedin.com/in/priyaiyer", date: "2026-04-22", today: false, sizeKB: 21 },
    { url: "linkedin.com/in/danielokonkwo", date: "2026-04-23", today: false, sizeKB: 18 },
    { url: "linkedin.com/in/sorayanadim", date: "2026-04-21", today: false, sizeKB: 19 },
    { url: "linkedin.com/in/mayapatel", date: "2026-04-26", today: false, sizeKB: 31, isUser: true },
  ];

  return (
    <div className="card" style={{overflow:"hidden"}}>
      <div style={{padding:"12px 16px", borderBottom:"1px solid var(--line-2)", display:"flex", justifyContent:"space-between", alignItems:"center"}}>
        <div>
          <div className="tag">CACHE INSPECTOR</div>
          <div style={{fontSize:12, color:"var(--ink-3)", marginTop:2}}>
            Confirms once-per-day rule · keyed by URL + date · older versions kept
          </div>
        </div>
        <input className="input" placeholder="Filter by URL…" style={{maxWidth:240, fontSize:13}}/>
      </div>
      <table className="feedback-table">
        <thead>
          <tr>
            <th>LinkedIn URL</th>
            <th style={{width:120}}>Snapshot date</th>
            <th style={{width:80}}>Status</th>
            <th style={{width:80}}>Size</th>
            <th style={{width:200}}></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} style={{background: r.today ? "var(--accent-soft)" : "transparent"}}>
              <td className="mono" style={{fontSize:12}}>
                {r.url} {r.isUser && <span style={{
                  fontSize:10, marginLeft:8, color:"var(--ink-3)",
                  border:"1px solid var(--line)", padding:"1px 6px", borderRadius:99,
                }}>user</span>}
              </td>
              <td className="mono" style={{fontSize:12}}>{r.date}</td>
              <td>
                {r.today
                  ? <span style={{fontSize:11, color:"var(--accent)"}}>● today</span>
                  : <span style={{fontSize:11, color:"var(--ink-3)"}}>archived</span>}
              </td>
              <td className="mono" style={{fontSize:11, color:"var(--ink-3)"}}>{r.sizeKB} KB</td>
              <td>
                <div style={{display:"flex", gap:6}}>
                  <button className="btn subtle sm">View JSON</button>
                  <button className="btn subtle sm" disabled={r.today} title={r.today ? "Already cached today — try tomorrow" : ""}>
                    Force refresh
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

window.Admin = Admin;
