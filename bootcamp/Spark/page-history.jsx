/* global React, SPARK_DATA */
// History page — search-first
function History({ density }) {
  const compact = density === "compact";
  const items = window.SPARK_DATA.HISTORY;
  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">wireframe · history · search-first · click row to re-open cached results</div>

      <div className="center-stack wide">
        <div style={{display:"flex", alignItems:"center", gap:10, marginBottom: compact ? 20 : 32}}>
          <div className="card" style={{flex:1, padding: 4, display:"flex", gap:6, alignItems:"center"}}>
            <span style={{padding:"0 10px", color:"var(--ink-3)"}}>⌕</span>
            <input
              className="input"
              placeholder="Search by name, company, or topic…"
              style={{border:"none", padding:"10px 4px"}}
            />
            <span className="kbd">⌘ K</span>
          </div>
        </div>

        <div className="tag" style={{marginBottom:12}}>Recent</div>

        <div className="card">
          {items.map((it, i) => (
            <a key={i} href="#" style={{
              display:"flex", alignItems:"center", gap:14,
              padding: compact ? "10px 16px" : "14px 18px",
              borderBottom: i < items.length - 1 ? "1px solid var(--line-2)" : "none",
              textDecoration: "none", color: "inherit",
            }}>
              <div className="avatar sm" />
              <div style={{flex:1, minWidth:0}}>
                <div style={{fontSize:14, color:"var(--ink)"}}>{it.name}</div>
                <div style={{fontSize:12, color:"var(--ink-3)"}}>{it.company}</div>
              </div>
              <span style={{fontSize:12, color:"var(--ink-3)"}}>looked up {it.when}</span>
              <span style={{color:"var(--ink-4)"}}>›</span>
            </a>
          ))}
        </div>

        {/* Empty state preview */}
        <div style={{marginTop: 40}}>
          <div className="page-note">empty state preview —</div>
          <div className="card" style={{padding: 40, textAlign:"center"}}>
            <p className="serif" style={{fontSize: 18, margin: "0 0 8px"}}>No lookups yet.</p>
            <p style={{color:"var(--ink-3)", margin: 0, fontSize: 13}}>Paste a LinkedIn URL to start.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

window.History = History;
