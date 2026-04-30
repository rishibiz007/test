"use client";
import { useState } from "react";
import { Avatar, BrandLink, Icon } from "./UI";
import type { AppState, UserProfile } from "@/lib/types";

interface Props {
  state: AppState;
  update: (patch: Partial<AppState>) => void;
  onDone: () => void;
}

function InlineField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  const [editing, setEditing] = useState(false);

  return (
    <div className="profile-row">
      <span className="label">{label}</span>
      {editing ? (
        <input
          className="input"
          style={{ flex: 1, fontSize: 13, padding: "2px 6px", height: 28 }}
          value={value}
          autoFocus
          onChange={(e) => onChange(e.target.value)}
          onBlur={() => setEditing(false)}
          onKeyDown={(e) => { if (e.key === "Enter") setEditing(false); }}
        />
      ) : (
        <span style={{ flex: 1, color: value ? "inherit" : "var(--ink-4)", fontStyle: value ? "normal" : "italic" }}>
          {value || "—"}
        </span>
      )}
      <button
        onClick={() => setEditing((v) => !v)}
        style={{ background: "none", border: 0, color: "var(--ink-4)", fontSize: 11, cursor: "pointer", padding: "0 2px" }}
      >
        {editing ? "done" : "edit"}
      </button>
    </div>
  );
}

export default function Onboarding({ state, update, onDone }: Props) {
  const [step, setStep] = useState(0);
  const [linkedin, setLinkedin] = useState(state.user.linkedin || "");
  const [fetching, setFetching] = useState(false);
  const [pulled, setPulled] = useState<UserProfile | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const [education, setEducation] = useState("");
  const [recentPosts, setRecentPosts] = useState("");
  const [podcasts, setPodcasts] = useState("");
  const [lookingFor, setLookingFor] = useState("");
  const [talksAbout, setTalksAbout] = useState("");

  const startFetch = async () => {
    if (!linkedin.trim()) return;
    setFetching(true);
    setFetchError(null);
    try {
      const res = await fetch("/api/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ handle: linkedin.trim() }),
      });
      const data = await res.json();
      if (!res.ok) {
        setFetchError(data.error ?? "Failed to fetch profile.");
        setFetching(false);
        return;
      }
      const fetched = data.user as UserProfile;
      setPulled(fetched);
      setEducation(fetched.education);
      setRecentPosts(fetched.recentPosts);
      setPodcasts(fetched.podcasts);
      setLookingFor(fetched.lookingFor);
      setTalksAbout(fetched.talksAbout);
      setStep(1);
    } catch {
      setFetchError("Network error — please try again.");
    } finally {
      setFetching(false);
    }
  };

  const finish = () => {
    if (!pulled) return;
    update({
      onboarded: true,
      user: { ...pulled, education, recentPosts, podcasts, lookingFor, talksAbout },
    });
    onDone();
  };

  return (
    <div className="onb-shell" data-screen-label="01 onboarding">
      <header style={{ padding: "20px 24px", display: "flex", alignItems: "center" }}>
        <BrandLink onClick={() => {}} />
        <div style={{ flex: 1 }} />
        <span className="muted" style={{ fontSize: 12 }}>
          First-run setup
        </span>
      </header>

      <div className="onb-card">
        <div className="onb-dots">
          <span className={step === 0 ? "active" : "done"} />
          <span className={step === 1 ? "active" : step > 1 ? "done" : ""} />
          <span className={step === 2 ? "active" : ""} />
        </div>

        {step === 0 && (
          <div className="onb-step-enter">
            <h1 className="h-1 serif" style={{ marginBottom: 8 }}>
              Let&apos;s start with you.
            </h1>
            <p className="muted" style={{ marginBottom: 28 }}>
              Paste your LinkedIn URL — we&apos;ll prefill the rest.
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
                {fetching ? "Reading…" : "Continue"} <Icon name="arrow" />
              </button>
            </div>
            {fetching && (
              <div style={{ marginTop: 28, display: "flex", flexDirection: "column", gap: 8 }}>
                <div className="skel" style={{ height: 12, width: "70%" }} />
                <div className="skel" style={{ height: 12, width: "55%" }} />
                <div className="skel" style={{ height: 12, width: "80%" }} />
              </div>
            )}
            {fetchError && (
              <p style={{ marginTop: 16, color: "var(--red, #c0392b)", fontSize: 13 }}>
                {fetchError}
              </p>
            )}
            <button
              onClick={() => setLinkedin("linkedin.com/in/mayapatel")}
              style={{ marginTop: 36, border: 0, background: "transparent", color: "var(--ink-4)", fontSize: 11, cursor: "pointer" }}
            >
              ⤴ use demo URL
            </button>
          </div>
        )}

        {step === 1 && pulled && (
          <div className="onb-step-enter">
            <h1 className="h-1 serif" style={{ marginBottom: 8 }}>
              Anything off?
            </h1>
            <p className="muted" style={{ marginBottom: 24 }}>
              Edit inline. The personalization is only as good as this.
            </p>
            <div className="card" style={{ padding: 18 }}>
              <div style={{ display: "flex", gap: 14, alignItems: "center", marginBottom: 14 }}>
                <Avatar initials={pulled.initials} size="lg" tone="amber" />
                <div>
                  <div style={{ fontSize: 15, fontWeight: 500 }}>{pulled.name}</div>
                  <div className="muted" style={{ fontSize: 13 }}>{pulled.role}</div>
                </div>
              </div>
              <InlineField label="Education" value={education} onChange={setEducation} />
              <InlineField label="Recent" value={recentPosts} onChange={setRecentPosts} />
              <InlineField label="Podcast" value={podcasts} onChange={setPodcasts} />
            </div>
            <div style={{ marginTop: 22 }}>
              <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>
                Looking for
              </label>
              <textarea
                className="textarea"
                placeholder="e.g. PM roles at Series B+ startups in SF or remote"
                value={lookingFor}
                onChange={(e) => setLookingFor(e.target.value)}
              />
            </div>
            <div style={{ marginTop: 16 }}>
              <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>
                Love talking about
              </label>
              <textarea
                className="textarea"
                placeholder="e.g. Climate tech, product strategy, coffee"
                value={talksAbout}
                onChange={(e) => setTalksAbout(e.target.value)}
              />
            </div>
            <div style={{ marginTop: 22, display: "flex", justifyContent: "space-between" }}>
              <button className="btn ghost" onClick={() => setStep(0)}>
                ← Back
              </button>
              <button className="btn" onClick={() => setStep(2)}>
                Looks right <Icon name="arrow" />
              </button>
            </div>
          </div>
        )}

        {step === 2 && pulled && (
          <div className="onb-step-enter" style={{ textAlign: "center" }}>
            <div
              style={{
                width: 56, height: 56, borderRadius: 999,
                background: "var(--accent-soft)", border: "1px solid var(--accent-line)",
                display: "inline-flex", alignItems: "center", justifyContent: "center",
                color: "var(--accent)", margin: "8px 0 22px",
              }}
            >
              <Icon name="check" size={20} />
            </div>
            <h1 className="h-1 serif" style={{ marginBottom: 10 }}>
              You&apos;re set, {pulled.name.split(" ")[0]}.
            </h1>
            <p className="muted" style={{ marginBottom: 32, maxWidth: 360, marginLeft: "auto", marginRight: "auto" }}>
              Spark will use what you just shared to personalize every lookup.
            </p>
            <button className="btn lg" onClick={finish}>
              Look up your first person <Icon name="arrow" />
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
