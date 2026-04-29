/* global React */
// Lookup / Home — 3 variants

function Lookup({ variant, density }) {
  const compact = density === "compact";
  if (variant === "split")  return <LookupSplit compact={compact} />;
  if (variant === "recents") return <LookupRecents compact={compact} />;
  return <LookupHero compact={compact} />;
}

/* ===================== A · Centered hero (briefed default) ===================== */
function LookupHero({ compact }) {
  return (
    <div className={`page ${compact ? "compact" : ""}`} style={{display:"flex", flexDirection:"column", justifyContent:"center", minHeight: compact ? "auto" : "70vh"}}>
      <div className="page-note">wireframe · lookup · variant: <strong style={{color:"var(--ink-2)"}}>A · Centered hero</strong></div>
      <div className="center-stack" style={{textAlign:"center"}}>
        <h1 className="h1 serif" style={{fontSize: compact ? 28 : 40, marginBottom: 14}}>
          Who are you meeting?
        </h1>
        <p className="lede" style={{marginBottom: compact ? 24 : 36}}>
          Paste a LinkedIn URL. Spark reads their profile, recent posts, and any podcast or blog appearances —
          then surfaces what you have in common.
        </p>

        <div className="card" style={{padding: 6, display:"flex", gap:6, marginBottom: 14, textAlign:"left"}}>
          <input
            className="input"
            placeholder="linkedin.com/in/…"
            style={{border:"none", padding:"12px 14px", fontFamily:"JetBrains Mono, monospace", fontSize:13}}
          />
          <button className="btn">Look up →</button>
        </div>

        <p style={{fontSize:12, color:"var(--ink-3)", marginTop:0, marginBottom: compact ? 24 : 40}}>
          We only use publicly available information. Personalized using your profile.
        </p>

        <div className="card" style={{padding: compact ? 14 : 18, textAlign:"left", maxWidth: 480, margin:"0 auto", background:"var(--bg-2)"}}>
          <div className="tag" style={{marginBottom:10}}>If you'd just submitted —</div>
          <ProgressLine done text="Reading her profile" />
          <ProgressLine done text="Checking recent posts (4 found)" />
          <ProgressLine active text="Scanning podcast & blog mentions…" />
          <ProgressLine text="Finding overlap with your profile" />
        </div>
      </div>
    </div>
  );
}

/* ===================== B · Split: input + "what we'll do" preview ===================== */
function LookupSplit({ compact }) {
  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">wireframe · lookup · variant: <strong style={{color:"var(--ink-2)"}}>B · Split — input left, preview right</strong></div>
      <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap: 48, alignItems:"center", maxWidth: 1100, margin: "0 auto", minHeight: compact ? "auto" : "60vh"}}>
        {/* Left: the actual ask */}
        <div>
          <div className="tag" style={{marginBottom:16}}>NEW LOOKUP</div>
          <h1 className="h1 serif" style={{fontSize: compact ? 30 : 38, marginBottom: 14, lineHeight:1.15}}>
            Paste a LinkedIn URL of someone you're meeting.
          </h1>
          <p className="lede" style={{marginBottom: 24}}>
            Spark uses your profile to find what you actually have in common — not surface trivia.
          </p>
          <div className="card" style={{padding: 6, display:"flex", gap:6, marginBottom: 12}}>
            <input
              className="input"
              placeholder="linkedin.com/in/…"
              style={{border:"none", padding:"12px 14px", fontFamily:"JetBrains Mono, monospace", fontSize:13}}
            />
            <button className="btn">Look up →</button>
          </div>
          <p style={{fontSize:11, color:"var(--ink-3)", margin:0, fontFamily:"JetBrains Mono, monospace"}}>
            public info only · personalized with your profile · cached 24h
          </p>
        </div>

        {/* Right: process preview / what we read */}
        <div className="card" style={{padding: compact ? 20 : 28, background:"var(--bg-2)"}}>
          <div className="tag" style={{marginBottom:14}}>WHAT WE'LL READ</div>
          <PreviewLine label="Profile" detail="role, education, work history" />
          <PreviewLine label="Recent posts" detail="last 90 days · LinkedIn + Substack" />
          <PreviewLine label="Podcast / blog" detail="if any guest appearances" />
          <PreviewLine label="Mutual overlap" detail="cross-referenced with your profile" />
          <hr className="divider" style={{margin:"18px 0"}} />
          <div className="tag" style={{marginBottom:10}}>WHAT WE'LL OUTPUT</div>
          <p className="serif" style={{fontSize:14, color:"var(--ink-2)", lineHeight:1.55, margin:0}}>
            3–5 topics, each with a one-sentence "why," a citation, and a try-saying line you can lift verbatim.
          </p>
        </div>
      </div>
    </div>
  );
}

function PreviewLine({ label, detail }) {
  return (
    <div style={{display:"flex", gap:12, padding:"7px 0", alignItems:"flex-start", borderBottom:"1px dashed var(--line-2)"}}>
      <span style={{
        width:14, height:14, borderRadius:99,
        border:"1px solid var(--ink-3)", marginTop: 3, flexShrink:0,
      }}/>
      <div style={{flex:1, minWidth:0}}>
        <div style={{fontSize:13, color:"var(--ink)"}}>{label}</div>
        <div style={{fontSize:11, color:"var(--ink-3)", fontFamily:"JetBrains Mono, monospace"}}>{detail}</div>
      </div>
    </div>
  );
}

/* ===================== C · Lookup + recents inline ===================== */
function LookupRecents({ compact }) {
  const recent = (window.SPARK_DATA && window.SPARK_DATA.HISTORY) || [];
  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">wireframe · lookup · variant: <strong style={{color:"var(--ink-2)"}}>C · Input + inline recents</strong> · power-user shape</div>
      <div className="center-stack wide">
        <h1 className="h1 serif" style={{fontSize: compact ? 26 : 32, marginBottom: 18, textAlign:"center"}}>
          Who are you meeting?
        </h1>

        <div className="card" style={{padding: 6, display:"flex", gap:6, marginBottom: 10}}>
          <span style={{padding:"0 12px", color:"var(--ink-3)", display:"flex", alignItems:"center"}}>↗</span>
          <input
            className="input"
            placeholder="Paste a LinkedIn URL · ⌘V"
            style={{border:"none", padding:"12px 4px", fontFamily:"JetBrains Mono, monospace", fontSize:13}}
          />
          <button className="btn">Look up →</button>
        </div>

        <p style={{fontSize:11, color:"var(--ink-3)", textAlign:"center", marginBottom: compact ? 28 : 44, fontFamily:"JetBrains Mono, monospace"}}>
          public info only · personalized using your profile
        </p>

        <div style={{display:"flex", justifyContent:"space-between", alignItems:"baseline", marginBottom:10}}>
          <span className="tag">RECENT</span>
          <a href="#" style={{fontSize:12, color:"var(--ink-3)", textDecoration:"underline", textDecorationColor:"var(--line)"}}>See all →</a>
        </div>

        <div className="card">
          {recent.slice(0, 4).map((it, i) => (
            <a key={i} href="#" style={{
              display:"flex", alignItems:"center", gap:14,
              padding: compact ? "10px 16px" : "14px 18px",
              borderBottom: i < Math.min(recent.length, 4) - 1 ? "1px solid var(--line-2)" : "none",
              textDecoration:"none", color:"inherit",
            }}>
              <div className="avatar sm" />
              <div style={{flex:1, minWidth:0}}>
                <div style={{fontSize:14, color:"var(--ink)"}}>{it.name}</div>
                <div style={{fontSize:12, color:"var(--ink-3)"}}>{it.company}</div>
              </div>
              <span style={{fontSize:11, color:"var(--ink-3)", fontFamily:"JetBrains Mono, monospace"}}>{it.when}</span>
              <span className="kbd" style={{fontSize:9}}>cached</span>
              <span style={{color:"var(--ink-4)"}}>›</span>
            </a>
          ))}
        </div>

        <div style={{marginTop: 18, fontSize:12, color:"var(--ink-3)", textAlign:"center"}}>
          Re-opening a recent person uses the cached snapshot · no re-scrape.
        </div>
      </div>
    </div>
  );
}

/* ===================== shared ===================== */
function ProgressLine({ done, active, text }) {
  return (
    <div style={{display:"flex", alignItems:"center", gap:10, padding:"5px 0", fontSize:13, color: done ? "var(--ink-3)" : active ? "var(--ink)" : "var(--ink-4)"}}>
      <span style={{
        width:10, height:10, borderRadius:99,
        border: "1px solid",
        borderColor: done ? "var(--ink-3)" : active ? "var(--accent)" : "var(--ink-4)",
        background: done ? "var(--ink-3)" : active ? "var(--accent)" : "transparent",
      }}/>
      <span style={{fontFamily: "JetBrains Mono, monospace", fontSize: 12}}>
        {done ? "✓ " : active ? "→ " : "  "}{text}
      </span>
    </div>
  );
}

window.Lookup = Lookup;
