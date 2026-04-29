/* global React */
// Onboarding — 3 variants

function Onboarding({ variant, density }) {
  const compact = density === "compact";
  if (variant === "stepped")  return <OnboardingStepped compact={compact} />;
  if (variant === "scroll")   return <OnboardingScroll compact={compact} />;
  return <OnboardingChat compact={compact} />;
}

/* ===================== A · Conversational chat ===================== */
function OnboardingChat({ compact }) {
  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">
        wireframe · onboarding · variant: <strong style={{color:"var(--ink-2)"}}>A · Conversational chat</strong>
      </div>
      <div className="center-stack" style={{maxWidth: 680}}>
        <div className="card" style={{padding: compact ? 20 : 28}}>
          <div style={{display:"flex", alignItems:"center", gap:8, marginBottom: compact ? 18 : 28}}>
            <span style={{width:8, height:8, borderRadius:99, background:"var(--ink)"}}/>
            <span style={{width:8, height:8, borderRadius:99, background:"var(--ink)"}}/>
            <span style={{width:8, height:8, borderRadius:99, background:"var(--ink-4)"}}/>
            <span className="tag" style={{marginLeft:"auto"}}>Step 2 of 3</span>
          </div>

          <div style={{display:"flex", flexDirection:"column", gap: compact ? 14 : 20}}>
            <ChatBubble side="bot">Hey — I'm Spark. I'll help you walk into your next coffee chat with something to actually say.</ChatBubble>
            <ChatBubble side="bot">First, paste a link to your LinkedIn so I can learn about you.</ChatBubble>
            <ChatBubble side="user">linkedin.com/in/mayapatel</ChatBubble>
            <ChatBubble side="bot">Got it. Here's what I pulled — anything off, edit it inline.</ChatBubble>

            <div className="card" style={{padding:18, background:"var(--bg-2)"}}>
              <div style={{display:"flex", gap:14, alignItems:"flex-start"}}>
                <div className="avatar md" />
                <div style={{flex:1, minWidth:0}}>
                  <EditableField label="Name" value="Maya Patel" />
                  <EditableField label="Current role" value="PM, ex-Notion · looking" />
                  <EditableField label="Education" value="UC Berkeley, B.S. Computer Science '13" />
                  <EditableField label="Recent post" value="Why I'm betting on climate tools for SMBs" />
                  <EditableField label="Podcast" value="Lenny's Podcast (guest, S4E12)" />
                </div>
              </div>
            </div>

            <ChatBubble side="bot">Two more — these are the ones the AI actually personalizes from. Be specific.</ChatBubble>

            <div style={{display:"flex", flexDirection:"column", gap:10}}>
              <label className="tag">What are you looking for right now?</label>
              <textarea className="textarea" defaultValue="PM roles at Series B+ climate / fintech startups in SF or remote." />
            </div>
            <div style={{display:"flex", flexDirection:"column", gap:10}}>
              <label className="tag">What do you love talking about?</label>
              <textarea className="textarea" defaultValue="Climate tech, Indian street food, Bay Area soccer leagues, post-Notion product taste." />
            </div>

            <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginTop:8}}>
              <button className="btn ghost subtle">← Back</button>
              <button className="btn">Looks right →</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ===================== B · Stepped (briefed default) =====================
   Three full-screen one-question-at-a-time steps shown side-by-side as frames. */
function OnboardingStepped({ compact }) {
  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">
        wireframe · onboarding · variant: <strong style={{color:"var(--ink-2)"}}>B · 3 stepped screens</strong> · briefed default · all 3 shown for review
      </div>
      <div className="variant-grid cols-3">
        <StepFrame n={1} title="Step 1 · Paste your LinkedIn">
          <div style={{padding: compact ? 24 : 40, textAlign:"center"}}>
            <Dots n={3} active={0} />
            <h2 className="serif" style={{fontSize: 22, margin: "32px 0 8px", lineHeight:1.3, fontWeight:400}}>
              Let's start with you.
            </h2>
            <p style={{color:"var(--ink-3)", margin:"0 0 28px", fontSize:13}}>
              Paste a link to your LinkedIn. We'll prefill the rest.
            </p>
            <input
              className="input"
              placeholder="linkedin.com/in/…"
              style={{textAlign:"center", fontFamily:"JetBrains Mono, monospace", fontSize:13, padding:"14px"}}
            />
            <div style={{marginTop: 22}}>
              <button className="btn" style={{width:"100%", justifyContent:"center"}}>Continue →</button>
            </div>
            <p style={{fontSize:11, color:"var(--ink-3)", marginTop:18, lineHeight:1.5}}>
              We only read public information.<br/>You can edit anything we pull on the next screen.
            </p>
          </div>
        </StepFrame>

        <StepFrame n={2} title="Step 2 · Confirm your profile">
          <div style={{padding: compact ? 18 : 28}}>
            <Dots n={3} active={1} />
            <h2 className="serif" style={{fontSize: 20, margin: "22px 0 16px", fontWeight:400}}>
              Does this look right?
            </h2>
            <div style={{display:"flex", gap:12, alignItems:"center", marginBottom:14}}>
              <div className="avatar md" />
              <div style={{flex:1, minWidth:0}}>
                <div style={{fontSize:14, fontWeight:500}}>Maya Patel</div>
                <div style={{fontSize:12, color:"var(--ink-3)"}}>PM, ex-Notion · looking</div>
              </div>
            </div>
            <EditableField label="Education" value="Berkeley CS '13" />
            <EditableField label="Recent post" value="Climate tools for SMBs" />
            <EditableField label="Podcast" value="Lenny's S4E12" />
            <div style={{marginTop:14}}>
              <label className="tag" style={{display:"block", marginBottom:6}}>Looking for</label>
              <textarea className="textarea" style={{minHeight:50, fontSize:12}} defaultValue="PM roles at Series B+ climate / fintech startups." />
              <label className="tag" style={{display:"block", margin:"12px 0 6px"}}>Love talking about</label>
              <textarea className="textarea" style={{minHeight:50, fontSize:12}} defaultValue="Climate tech, chaat, soccer leagues." />
            </div>
            <div style={{display:"flex", gap:6, marginTop:16}}>
              <button className="btn subtle sm" style={{flex:1, justifyContent:"center"}}>← Back</button>
              <button className="btn sm" style={{flex:2, justifyContent:"center"}}>Looks right →</button>
            </div>
          </div>
        </StepFrame>

        <StepFrame n={3} title="Step 3 · You're set">
          <div style={{padding: compact ? 24 : 40, textAlign:"center"}}>
            <Dots n={3} active={2} />
            <div style={{
              width:48, height:48, borderRadius:99, background:"var(--accent-soft)",
              border:"1px solid color-mix(in oklch, var(--accent) 40%, var(--line))",
              display:"flex", alignItems:"center", justifyContent:"center",
              margin:"32px auto 18px", color:"var(--accent)", fontSize:20,
            }}>✓</div>
            <h2 className="serif" style={{fontSize: 22, margin: "0 0 10px", fontWeight:400}}>
              You're set, Maya.
            </h2>
            <p style={{color:"var(--ink-3)", margin:"0 0 28px", fontSize:13, lineHeight:1.6}}>
              Spark will use what you just shared to personalize every lookup.
            </p>
            <button className="btn" style={{width:"100%", justifyContent:"center"}}>Look up your first person →</button>
            <p style={{fontSize:11, color:"var(--ink-3)", marginTop:18}}>
              You can edit your profile anytime from the avatar menu.
            </p>
          </div>
        </StepFrame>
      </div>
    </div>
  );
}

function StepFrame({ n, title, children }) {
  return (
    <div className="variant-frame">
      <div className="variant-frame-head">
        <span>{title}</span>
        <span style={{color:"var(--ink-3)"}}>{n} / 3</span>
      </div>
      {children}
    </div>
  );
}

function Dots({ n, active }) {
  return (
    <div style={{display:"flex", justifyContent:"center", gap:6}}>
      {[...Array(n)].map((_, i) => (
        <span key={i} style={{
          width:7, height:7, borderRadius:99,
          background: i <= active ? "var(--ink)" : "var(--ink-4)",
        }}/>
      ))}
    </div>
  );
}

/* ===================== C · Single long scroll ===================== */
function OnboardingScroll({ compact }) {
  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">
        wireframe · onboarding · variant: <strong style={{color:"var(--ink-2)"}}>C · Single long scroll</strong> · whole shape visible at once · sticky progress rail
      </div>
      <div style={{display:"grid", gridTemplateColumns:"160px 1fr", gap: 40, maxWidth: 880, margin:"0 auto"}}>
        {/* Sticky progress rail */}
        <aside style={{position:"sticky", top: 80, alignSelf:"start"}}>
          <div className="tag" style={{marginBottom:14}}>SETUP</div>
          {[
            { n: 1, label: "Your LinkedIn", done: true },
            { n: 2, label: "Confirm profile", done: false, active: true },
            { n: 3, label: "Done", done: false },
          ].map(s => (
            <div key={s.n} style={{
              display:"flex", gap:10, padding:"8px 0", alignItems:"center",
              color: s.active ? "var(--ink)" : s.done ? "var(--ink-3)" : "var(--ink-4)",
              fontSize: 13,
            }}>
              <span style={{
                width:18, height:18, borderRadius:99,
                border:"1px solid",
                borderColor: s.active ? "var(--ink)" : s.done ? "var(--ink-3)" : "var(--ink-4)",
                background: s.done ? "var(--ink-3)" : "transparent",
                color: "var(--bg)",
                display:"inline-flex", alignItems:"center", justifyContent:"center",
                fontSize: 10,
              }}>{s.done ? "✓" : s.n}</span>
              {s.label}
            </div>
          ))}
        </aside>

        {/* Scroll body */}
        <div style={{display:"flex", flexDirection:"column", gap: compact ? 28 : 44}}>
          <header>
            <h1 className="h1 serif">Welcome to Spark.</h1>
            <p className="lede">Three short sections. We'll prefill what we can.</p>
          </header>

          <section>
            <div className="tag" style={{marginBottom:8}}>01 — YOUR LINKEDIN</div>
            <h2 className="h2 serif">Where do you live online?</h2>
            <input
              className="input"
              defaultValue="linkedin.com/in/mayapatel"
              style={{fontFamily:"JetBrains Mono, monospace", fontSize:13}}
            />
            <div style={{fontSize:12, color:"var(--accent)", marginTop:8, fontFamily:"JetBrains Mono, monospace"}}>
              ✓ Pulled · 5 fields prefilled below
            </div>
          </section>

          <section>
            <div className="tag" style={{marginBottom:8}}>02 — CONFIRM YOUR PROFILE</div>
            <h2 className="h2 serif">Anything off?</h2>
            <p style={{color:"var(--ink-3)", marginTop:0}}>Edit inline. The personalization is only as good as this.</p>
            <div className="card" style={{padding:18, marginTop:14}}>
              <EditableField label="Name" value="Maya Patel" />
              <EditableField label="Current role" value="PM, ex-Notion · looking" />
              <EditableField label="Education" value="UC Berkeley, B.S. Computer Science '13" />
              <EditableField label="Recent post" value="Why I'm betting on climate tools for SMBs" />
              <EditableField label="Podcast" value="Lenny's Podcast (S4E12)" />
            </div>

            <div style={{marginTop: 22}}>
              <label className="tag" style={{display:"block", marginBottom:8}}>What are you looking for right now?</label>
              <textarea className="textarea" defaultValue="PM roles at Series B+ climate / fintech startups in SF or remote." />
            </div>
            <div style={{marginTop: 18}}>
              <label className="tag" style={{display:"block", marginBottom:8}}>What do you love talking about?</label>
              <textarea className="textarea" defaultValue="Climate tech, Indian street food, Bay Area soccer leagues, post-Notion product taste." />
            </div>
          </section>

          <section>
            <div className="tag" style={{marginBottom:8}}>03 — YOU'RE SET</div>
            <h2 className="h2 serif">That's it.</h2>
            <p style={{color:"var(--ink-3)"}}>You can edit any of this from your profile later.</p>
            <button className="btn" style={{marginTop:10}}>Look up your first person →</button>
          </section>
        </div>
      </div>
    </div>
  );
}

/* ===================== shared bits ===================== */
function ChatBubble({ side, children }) {
  const isBot = side === "bot";
  return (
    <div style={{display:"flex", justifyContent: isBot ? "flex-start" : "flex-end"}}>
      <div style={{
        maxWidth: "80%",
        background: isBot ? "var(--bg-2)" : "var(--ink)",
        color: isBot ? "var(--ink)" : "var(--bg)",
        padding: "10px 14px",
        borderRadius: 8,
        fontSize: 14, lineHeight: 1.5,
      }}>
        {children}
      </div>
    </div>
  );
}

function EditableField({ label, value }) {
  return (
    <div style={{display:"flex", gap:12, padding:"7px 0", borderBottom:"1px dashed var(--line-2)"}}>
      <span className="tag" style={{minWidth:90}}>{label}</span>
      <span style={{flex:1, color:"var(--ink)", fontSize:13}}>{value}</span>
      <span className="mono" style={{fontSize:11, color:"var(--ink-3)"}}>edit</span>
    </div>
  );
}

window.Onboarding = Onboarding;
