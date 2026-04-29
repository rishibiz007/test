/* global React */
// Onboarding — refined stepped variant. Runs end-to-end with simulated LinkedIn fetch.
const { useState: useOnbState, useEffect: useOnbEffect } = React;

function Onboarding({ store, onDone }) {
  const [step, setStep] = useOnbState(0);
  const [linkedin, setLinkedin] = useOnbState(store.user.linkedin || "");
  const [fetching, setFetching] = useOnbState(false);
  const [pulled, setPulled] = useOnbState(null);
  const [lookingFor, setLookingFor] = useOnbState(store.user.lookingFor);
  const [talksAbout, setTalksAbout] = useOnbState(store.user.talksAbout);

  const startFetch = () => {
    if (!linkedin.trim()) return;
    setFetching(true);
    setTimeout(() => {
      setPulled({
        ...window.SparkData.DEFAULT_USER,
        linkedin: linkedin.trim(),
      });
      setFetching(false);
      setStep(1);
    }, 1100);
  };

  const finish = () => {
    store.update({
      onboarded: true,
      user: { ...store.user, ...pulled, lookingFor, talksAbout },
    });
    onDone();
  };

  return (
    <div className="onb-shell" data-screen-label="01 onboarding">
      <header style={{ padding: "20px 24px", display: "flex", alignItems: "center" }}>
        <window.UI.BrandLink onClick={() => {}} />
        <div style={{ flex: 1 }} />
        <span className="muted" style={{ fontSize: 12 }}>First-run setup</span>
      </header>

      <div className="onb-card">
        <div className="onb-dots">
          <span className={step === 0 ? "active" : "done"} />
          <span className={step === 1 ? "active" : step > 1 ? "done" : ""} />
          <span className={step === 2 ? "active" : ""} />
        </div>

        {step === 0 && (
          <div className="onb-step-enter">
            <h1 className="h-1 serif" style={{ marginBottom: 8 }}>Let's start with you.</h1>
            <p className="muted" style={{ marginBottom: 28 }}>
              Paste your LinkedIn URL — we'll prefill the rest.
            </p>
            <input
              className="input mono"
              placeholder="linkedin.com/in/…"
              value={linkedin}
              onChange={(e) => setLinkedin(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") startFetch(); }}
              autoFocus
            />
            <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span className="muted-2" style={{ fontSize: 11 }}>
                We only read public information.
              </span>
              <button className="btn" onClick={startFetch} disabled={!linkedin.trim() || fetching}>
                {fetching ? "Reading…" : "Continue"} <window.UI.Icon name="arrow" />
              </button>
            </div>
            {fetching && (
              <div style={{ marginTop: 28, display: "flex", flexDirection: "column", gap: 8 }}>
                <div className="skel" style={{ height: 12, width: "70%" }} />
                <div className="skel" style={{ height: 12, width: "55%" }} />
                <div className="skel" style={{ height: 12, width: "80%" }} />
              </div>
            )}
            <button
              onClick={() => { setLinkedin("linkedin.com/in/mayapatel"); }}
              style={{ marginTop: 36, border: "0", background: "transparent", color: "var(--ink-4)", fontSize: 11, cursor: "pointer" }}>
              ⤴ use demo URL
            </button>
          </div>
        )}

        {step === 1 && pulled && (
          <div className="onb-step-enter">
            <h1 className="h-1 serif" style={{ marginBottom: 8 }}>Anything off?</h1>
            <p className="muted" style={{ marginBottom: 24 }}>
              Edit inline. The personalization is only as good as this.
            </p>
            <div className="card" style={{ padding: 18 }}>
              <div style={{ display: "flex", gap: 14, alignItems: "center", marginBottom: 14 }}>
                <window.UI.Avatar initials={pulled.initials} size="lg" tone="amber" />
                <div>
                  <div style={{ fontSize: 15, fontWeight: 500 }}>{pulled.name}</div>
                  <div className="muted" style={{ fontSize: 13 }}>{pulled.role}</div>
                </div>
              </div>
              <div className="profile-row"><span className="label">Education</span><span>{pulled.education}</span><span className="muted-2" style={{fontSize:11}}>edit</span></div>
              <div className="profile-row"><span className="label">Recent</span><span>{pulled.recentPosts.split(" · ")[0]}</span><span className="muted-2" style={{fontSize:11}}>edit</span></div>
              <div className="profile-row"><span className="label">Podcast</span><span>{pulled.podcasts}</span><span className="muted-2" style={{fontSize:11}}>edit</span></div>
            </div>
            <div style={{ marginTop: 22 }}>
              <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>Looking for</label>
              <textarea className="textarea" value={lookingFor} onChange={(e) => setLookingFor(e.target.value)} />
            </div>
            <div style={{ marginTop: 16 }}>
              <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>Love talking about</label>
              <textarea className="textarea" value={talksAbout} onChange={(e) => setTalksAbout(e.target.value)} />
            </div>
            <div style={{ marginTop: 22, display: "flex", justifyContent: "space-between" }}>
              <button className="btn ghost" onClick={() => setStep(0)}>← Back</button>
              <button className="btn" onClick={() => setStep(2)}>Looks right <window.UI.Icon name="arrow" /></button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="onb-step-enter" style={{ textAlign: "center" }}>
            <div style={{
              width: 56, height: 56, borderRadius: 999,
              background: "var(--accent-soft)", border: "1px solid var(--accent-line)",
              display: "inline-flex", alignItems: "center", justifyContent: "center",
              color: "var(--accent)", margin: "8px 0 22px",
            }}>
              <window.UI.Icon name="check" size={20} />
            </div>
            <h1 className="h-1 serif" style={{ marginBottom: 10 }}>You're set, {pulled.name.split(" ")[0]}.</h1>
            <p className="muted" style={{ marginBottom: 32, maxWidth: 360, marginLeft: "auto", marginRight: "auto" }}>
              Spark will use what you just shared to personalize every lookup.
            </p>
            <button className="btn lg" onClick={finish}>
              Look up your first person <window.UI.Icon name="arrow" />
            </button>
            <p className="muted-2" style={{ fontSize: 11, marginTop: 22 }}>
              You can edit your profile anytime from the avatar menu.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

window.Onboarding = Onboarding;
