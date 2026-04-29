/* global React, SPARK_DATA */
// Results page — 3 variants
const { useState: useResultsState } = React;

function Results({ variant, density }) {
  const compact = density === "compact";
  const T = window.SPARK_DATA.TOPICS;
  const target = window.SPARK_DATA.TARGET;

  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">
        wireframe · results · variant: <strong style={{color:"var(--ink-2)"}}>{variantLabel(variant)}</strong>
      </div>

      {variant === "stacked" && <ResultsStacked target={target} topics={T} compact={compact} />}
      {variant === "notecards" && <ResultsNotecards target={target} topics={T} compact={compact} />}
      {variant === "editorial" && <ResultsEditorial target={target} topics={T} compact={compact} />}
    </div>
  );
}

function variantLabel(v) {
  return {
    stacked: "Stacked cards (briefed default) — Tag → big serif quote → source",
    notecards: "Notecard stack — source-first, evidence leads",
    editorial: "Magazine column — pull-quote 'try saying' openers",
  }[v];
}

/* ---------- Header (shared) ---------- */
function ResultsHeader({ target }) {
  return (
    <div className="center-stack wide" style={{marginBottom: 20}}>
      <div style={{display:"flex", gap:14, alignItems:"center"}}>
        <div className="avatar md" />
        <div style={{flex:1, minWidth:0}}>
          <div className="serif" style={{fontSize: 20, lineHeight: 1.2}}>{target.name}</div>
          <div style={{color:"var(--ink-3)", fontSize: 13, marginTop: 2}}>
            {target.role} · {target.company}
          </div>
        </div>
        <span className="mono" style={{fontSize:11, color:"var(--ink-3)"}}>linkedin.com/in/priyaiyer</span>
      </div>
      <div style={{display:"flex", alignItems:"center", gap:8, marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--line-2)"}}>
        <span style={{fontSize:12, color:"var(--ink-2)"}}>Personalized for you</span>
        <span style={{
          fontSize:10, color:"var(--ink-3)",
          border:"1px solid var(--line)", borderRadius: 99,
          width: 14, height: 14, display:"inline-flex", alignItems:"center", justifyContent:"center",
        }} title="Topics use overlap between your profile and theirs.">i</span>
        <span className="mono" style={{fontSize:11, color:"var(--ink-3)", marginLeft:"auto"}}>· using today's snapshot</span>
      </div>
    </div>
  );
}

/* ---------- Variant A: Stacked cards ---------- */
function ResultsStacked({ target, topics, compact }) {
  return (
    <>
      <ResultsHeader target={target} />
      <div className="center-stack wide" style={{display:"flex", flexDirection:"column", gap: compact ? 14 : 20}}>
        {topics.map((t, i) => (
          <StackedCard key={t.id} t={t} idx={i} compact={compact} />
        ))}
      </div>
    </>
  );
}

function StackedCard({ t, idx, compact }) {
  // Card 0 = liked (accent border), Card 2 = downvoted (panel open + dimmed), others neutral
  const liked = idx === 0;
  const downed = idx === 2;
  return (
    <div className="card" style={{
      padding: compact ? 18 : 24,
      borderLeft: liked ? "3px solid var(--accent)" : "1px solid var(--line)",
      opacity: downed ? 0.6 : 1,
    }}>
      <div style={{display:"flex", justifyContent:"space-between", alignItems:"flex-start", gap:14}}>
        <div style={{flex:1, minWidth:0}}>
          <div style={{display:"flex", gap:10, alignItems:"center", marginBottom:10}}>
            <span className="tag">{t.category}</span>
            {t.usedYou && <BasedOnYouBadge />}
          </div>
          <p className="serif" style={{
            fontSize: compact ? 17 : 19,
            lineHeight: 1.45,
            margin: 0,
            color: "var(--ink)",
          }}>
            {t.text}
          </p>
          <div style={{display:"flex", alignItems:"center", gap:6, marginTop: 14, fontSize:12, color:"var(--ink-3)"}}>
            <span>↗</span>
            <a href="#" style={{color:"var(--ink-2)", textDecoration:"underline", textDecorationColor:"var(--line)"}}>{t.source}</a>
          </div>
        </div>

        <CardActions liked={liked} downed={downed} />
      </div>

      {downed && <DownvotePanel />}
    </div>
  );
}

function CardActions({ liked, downed }) {
  return (
    <div style={{display:"flex", flexDirection:"column", gap:6}}>
      <IconBtn label="👍" filled={liked} />
      <IconBtn label="👎" filled={downed} />
      <IconBtn label="copy" small />
    </div>
  );
}

function IconBtn({ label, filled, small }) {
  return (
    <button style={{
      border: "1px solid var(--line)",
      background: filled ? "var(--ink)" : "var(--bg)",
      color: filled ? "var(--bg)" : "var(--ink-2)",
      width: 32, height: 32,
      borderRadius: 4,
      cursor: "pointer",
      fontSize: small ? 10 : 13,
      fontFamily: small ? "JetBrains Mono, monospace" : "inherit",
    }}>
      {label}
    </button>
  );
}

function BasedOnYouBadge() {
  return (
    <span style={{
      fontSize: 10,
      letterSpacing: 0.06,
      color: "var(--accent)",
      border: "1px solid color-mix(in oklch, var(--accent) 40%, var(--line))",
      background: "var(--accent-soft)",
      padding: "2px 8px",
      borderRadius: 99,
      fontFamily: "JetBrains Mono, monospace",
    }}>
      based on your profile
    </span>
  );
}

function DownvotePanel() {
  const reasons = ["Too generic", "Feels weird", "Not accurate", "Boring", "Wrong tone", "Outdated"];
  const selected = ["Too generic", "Boring"];
  return (
    <div style={{
      marginTop: 18,
      paddingTop: 18,
      borderTop: "1px dashed var(--line)",
    }}>
      <div className="tag" style={{marginBottom:10}}>What was off?</div>
      <div style={{display:"flex", flexWrap:"wrap", gap:6, marginBottom:14}}>
        {reasons.map(r => (
          <span key={r} style={{
            fontSize: 12,
            padding: "5px 10px",
            borderRadius: 99,
            border: "1px solid var(--line)",
            background: selected.includes(r) ? "var(--ink)" : "var(--bg)",
            color: selected.includes(r) ? "var(--bg)" : "var(--ink-2)",
            cursor: "pointer",
          }}>{r}</span>
        ))}
      </div>
      <input
        className="input"
        placeholder="Too résumé-ish — I want something about her, not her job"
        style={{fontSize:13}}
      />
      <div style={{display:"flex", justifyContent:"flex-end", marginTop:10}}>
        <button className="btn sm">Submit</button>
      </div>
    </div>
  );
}

/* ---------- Variant B: Notecards (source-first, evidence leads) ---------- */
function ResultsNotecards({ target, topics, compact }) {
  return (
    <>
      <ResultsHeader target={target} />
      <div className="center-stack wide" style={{display:"flex", flexDirection:"column", gap: compact ? 16 : 22}}>
        {topics.map((t, i) => (
          <Notecard key={t.id} t={t} idx={i} compact={compact} />
        ))}
      </div>
    </>
  );
}

function Notecard({ t, idx, compact }) {
  return (
    <div style={{
      display:"grid",
      gridTemplateColumns: "180px 1fr auto",
      gap: 20,
      padding: compact ? 16 : 22,
      background: idx % 2 === 0 ? "var(--bg)" : "var(--bg-2)",
      border: "1px solid var(--line-2)",
      borderRadius: 4,
    }}>
      {/* Left: source as the lede */}
      <div style={{borderRight:"1px dashed var(--line)", paddingRight: 14}}>
        <div className="tag" style={{marginBottom:6}}>SOURCE</div>
        <div className="mono" style={{fontSize:11, color:"var(--ink-2)", lineHeight:1.5}}>{t.source}</div>
        {t.usedYou && (
          <div style={{marginTop:10}}>
            <BasedOnYouBadge />
          </div>
        )}
      </div>

      {/* Middle: evidence → conclusion */}
      <div>
        <div className="tag" style={{marginBottom: 8, color:"var(--ink-3)"}}>{t.category}</div>
        <p className="serif" style={{
          fontSize: compact ? 16 : 18,
          lineHeight: 1.5,
          margin: 0,
          color: "var(--ink)",
        }}>
          {t.text}
        </p>
        <div style={{
          marginTop: 12,
          paddingTop: 12,
          borderTop: "1px dashed var(--line)",
          fontSize: 13,
          color: "var(--ink-2)",
          fontStyle: "italic",
          fontFamily: "Source Serif 4, Georgia, serif",
        }}>
          → so try: <span style={{color:"var(--ink)"}}>{t.starter}</span>
        </div>
      </div>

      <CardActions liked={idx === 1} />
    </div>
  );
}

/* ---------- Variant C: Editorial / magazine column ---------- */
function ResultsEditorial({ target, topics, compact }) {
  return (
    <>
      <div className="center-stack" style={{maxWidth: 720, marginBottom: 24}}>
        <div className="tag" style={{marginBottom:14}}>COFFEE CHAT BRIEF · OCT 28, 2026</div>
        <h1 className="h1 serif" style={{fontSize: compact ? 32 : 40, marginBottom:6}}>{target.name}</h1>
        <div style={{color:"var(--ink-3)", fontSize:14, marginBottom:18}}>
          {target.role} at {target.company} · {target.bio.split("·").slice(2).join("·").trim()}
        </div>
        <hr className="divider" />
        <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", padding:"10px 0", fontSize:12, color:"var(--ink-3)"}}>
          <span>Personalized using your profile</span>
          <span className="mono">{topics.length} topics</span>
        </div>
        <hr className="divider" />
      </div>

      <div className="center-stack" style={{maxWidth: 720, display:"flex", flexDirection:"column", gap: compact ? 28 : 44}}>
        {topics.map((t, i) => (
          <EditorialTopic key={t.id} t={t} n={i+1} compact={compact} />
        ))}
      </div>
    </>
  );
}

function EditorialTopic({ t, n, compact }) {
  return (
    <article style={{position:"relative"}}>
      {/* Numeric marginalia */}
      <div style={{
        position:"absolute", left: -40, top: 6,
        fontFamily: "Source Serif 4, Georgia, serif",
        fontSize: 24, color: "var(--ink-4)",
      }}>
        {String(n).padStart(2,"0")}
      </div>

      <div style={{display:"flex", alignItems:"center", gap:10, marginBottom: 10}}>
        <span className="tag">{t.category}</span>
        {t.usedYou && <BasedOnYouBadge />}
        <div style={{flex:1}}/>
        <CardActions liked={n === 2} />
      </div>

      {/* Pull-quote try-saying line */}
      <blockquote className="serif" style={{
        margin: 0,
        padding: "8px 0 12px",
        fontSize: compact ? 19 : 22,
        lineHeight: 1.35,
        color: "var(--ink)",
        fontStyle: "italic",
        borderLeft: "2px solid var(--ink)",
        paddingLeft: 18,
      }}>
        "{t.starter}"
      </blockquote>

      <p style={{
        fontSize: 14,
        color: "var(--ink-2)",
        lineHeight: 1.6,
        margin: "14px 0 10px",
      }}>
        <span style={{color:"var(--ink-3)", fontFamily:"JetBrains Mono, monospace", fontSize:11, marginRight:8}}>WHY:</span>
        {t.text}
      </p>

      <div style={{fontSize:12, color:"var(--ink-3)", display:"flex", gap:8, alignItems:"center"}}>
        <span>↗</span>
        <a href="#" style={{color:"var(--ink-3)", textDecoration:"underline", textDecorationColor:"var(--line)"}}>{t.source}</a>
        {t.sourceUrl && <span className="mono" style={{fontSize:10, color:"var(--ink-4)"}}>· {t.sourceUrl}</span>}
      </div>
    </article>
  );
}

window.Results = Results;
