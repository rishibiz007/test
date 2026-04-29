/* global React */
// Reusable UI primitives for the Spark prototype.
const { useEffect, useRef, useState } = React;

function Avatar({ initials, size = "md", tone }) {
  const styleByTone = tone === "amber" ? {
    background: "linear-gradient(135deg, #fbf5e6 0%, #f6e7bd 100%)",
    color: "#8a6420",
    borderColor: "#ecd9a4",
  } : null;
  return (
    <span className={`avatar ${size}`} style={styleByTone || undefined}>
      {initials}
    </span>
  );
}

function BrandLink({ onClick }) {
  return (
    <a href="#" className="brand-link" onClick={(e) => { e.preventDefault(); onClick(); }}>
      <span className="brand-mark" />
      <span className="brand-name">Spark</span>
    </a>
  );
}

function AvatarMenu({ user, onProfile, onRerunOnboarding, onAdmin, onSignOut }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return;
    const close = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button className="avatar-menu-trigger" onClick={() => setOpen(o => !o)}>
        <Avatar initials={user.initials} size="sm" />
        <span className="name">{user.name.split(" ")[0]}</span>
        <span className="caret">▾</span>
      </button>
      {open && (
        <div className="dropdown" role="menu">
          <div className="head">
            <div style={{ fontSize: 13, color: "var(--ink)" }}>{user.name}</div>
            <div style={{ fontSize: 11, color: "var(--ink-3)" }}>{user.email}</div>
          </div>
          <button className="dropdown-item" onClick={() => { setOpen(false); onProfile(); }}>
            <Icon name="user" /> Your profile
          </button>
          <button className="dropdown-item" onClick={() => { setOpen(false); onRerunOnboarding(); }}>
            <Icon name="refresh" /> Re-run onboarding
          </button>
          <hr className="dropdown-sep" />
          <button className="dropdown-item" onClick={() => { setOpen(false); onAdmin(); }}>
            <Icon name="shield" /> Admin <span style={{ marginLeft: "auto", fontSize: 10, fontFamily: "JetBrains Mono, monospace", color: "var(--ink-4)" }}>internal</span>
          </button>
          <hr className="dropdown-sep" />
          <button className="dropdown-item muted" onClick={() => { setOpen(false); onSignOut(); }}>
            <Icon name="logout" /> Sign out
          </button>
        </div>
      )}
    </div>
  );
}

function Icon({ name, size = 14 }) {
  const paths = {
    user:    <><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/></>,
    refresh: <><path d="M3 12a9 9 0 0 1 15.5-6.3L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.3L3 16"/><path d="M3 21v-5h5"/></>,
    logout:  <><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5"/><path d="M21 12H9"/></>,
    shield:  <><path d="M12 3l8 3v6c0 4.5-3.5 8.5-8 9-4.5-.5-8-4.5-8-9V6l8-3z"/></>,
    search:  <><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></>,
    arrow:   <><path d="M5 12h14"/><path d="M13 5l7 7-7 7"/></>,
    up:      <><path d="M7 14l5-5 5 5"/></>,
    thumbsUp:   <><path d="M7 10v11"/><path d="M7 10l4-7c1.5 0 2.5 1 2.5 2.5V10h5a2 2 0 0 1 2 2.3l-1.4 7A2 2 0 0 1 17 21H7"/></>,
    thumbsDown: <><path d="M17 14V3"/><path d="M17 14l-4 7c-1.5 0-2.5-1-2.5-2.5V14h-5a2 2 0 0 1-2-2.3L5 4.7A2 2 0 0 1 7 3h10"/></>,
    copy:    <><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></>,
    chevron: <><path d="M9 6l6 6-6 6"/></>,
    check:   <><path d="M5 12l4 4 10-10"/></>,
    x:       <><path d="M6 6l12 12"/><path d="M18 6L6 18"/></>,
    extLink: <><path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6"/></>,
    spark:   <><path d="M12 2v6"/><path d="M12 16v6"/><path d="M2 12h6"/><path d="M16 12h6"/></>,
    plus:    <><path d="M12 5v14"/><path d="M5 12h14"/></>,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

function Toast({ text, actionLabel, onAction, onDismiss, ttl = 6000 }) {
  useEffect(() => {
    if (!ttl) return;
    const t = setTimeout(onDismiss, ttl);
    return () => clearTimeout(t);
  }, [ttl, onDismiss]);
  return (
    <div className="toast">
      <span>{text}</span>
      {actionLabel && <button onClick={onAction}>{actionLabel}</button>}
    </div>
  );
}

window.UI = { Avatar, BrandLink, AvatarMenu, Icon, Toast };
