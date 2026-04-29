/* global React */
// Profile page — single-variant
function Profile({ density }) {
  const compact = density === "compact";
  return (
    <div className={`page ${compact ? "compact" : ""}`}>
      <div className="page-note">
        wireframe · profile · auto-save on blur · refresh disabled if cached today
      </div>
      <div className="center-stack wide">
        <h1 className="h1 serif">Your profile</h1>
        <p className="lede">Used to personalize every lookup. Edits save automatically.</p>

        <div className="callout" style={{marginBottom: 24, justifyContent:"space-between"}}>
          <span>Last refreshed from LinkedIn — <strong>2 days ago</strong>. Today's snapshot already cached.</span>
          <button className="btn subtle sm" disabled title="Already refreshed today — try again tomorrow.">
            Refresh now
          </button>
        </div>

        <div className="card" style={{padding: compact ? 18 : 28}}>
          <div style={{display:"flex", gap:18, alignItems:"flex-start", marginBottom: 22}}>
            <div className="avatar lg" />
            <div style={{flex:1}}>
              <ProfileField label="Name" value="Maya Patel" />
              <ProfileField label="Current role" value="PM, ex-Notion · looking" />
              <ProfileField label="LinkedIn URL" value="linkedin.com/in/mayapatel" mono />
              <ProfileField label="Email" value="maya.p@example.com" />
            </div>
          </div>

          <hr className="divider" />
          <div style={{padding:"18px 0"}}>
            <ProfileField label="Education" value="UC Berkeley, B.S. Computer Science '13" />
            <ProfileField label="Recent posts" value="Why I'm betting on climate tools for SMBs · Notes from 6 months of unstructured time" multi />
            <ProfileField label="Podcasts / blog" value="Lenny's Podcast (guest, S4E12)" />
          </div>

          <hr className="divider" />
          <div style={{padding:"18px 0 0"}}>
            <label className="tag" style={{display:"block", marginBottom:8}}>What are you looking for right now?</label>
            <textarea className="textarea" defaultValue="PM roles at Series B+ climate / fintech startups in SF or remote." />
            <label className="tag" style={{display:"block", margin:"18px 0 8px"}}>What do you love talking about?</label>
            <textarea className="textarea" defaultValue="Climate tech, Indian street food, Bay Area soccer leagues, post-Notion product taste." />
          </div>
        </div>
      </div>
    </div>
  );
}

function ProfileField({ label, value, mono, multi }) {
  return (
    <div style={{display:"flex", gap:14, padding: multi ? "10px 0" : "8px 0", alignItems: multi ? "flex-start" : "center"}}>
      <span className="tag" style={{minWidth: 120, paddingTop: multi ? 4 : 0}}>{label}</span>
      <span style={{flex:1, fontFamily: mono ? "JetBrains Mono, monospace" : "inherit", fontSize: mono ? 13 : 14}}>
        {value}
      </span>
      <span className="mono" style={{fontSize:11, color:"var(--ink-3)"}}>edit</span>
    </div>
  );
}

window.Profile = Profile;
