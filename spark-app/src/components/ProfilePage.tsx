"use client";
import { Avatar } from "./UI";
import type { AppState } from "@/lib/types";

interface Props {
  state: AppState;
  update: (patch: Partial<AppState>) => void;
}

export default function ProfilePage({ state, update }: Props) {
  const u = state.user;

  const setField = (key: keyof typeof u, value: string) => {
    update({ user: { ...u, [key]: value } });
  };

  return (
    <div className="page narrow fade-in" data-screen-label="05 profile">
      <span className="eyebrow">YOUR PROFILE</span>
      <h1 className="h-1 serif" style={{ marginTop: 8, marginBottom: 22 }}>
        How Spark sees you
      </h1>

      <div
        className="card"
        style={{
          padding: "12px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "var(--bg-2)",
          marginBottom: 22,
          fontSize: 13,
        }}
      >
        <span className="muted">
          Last refreshed from LinkedIn —{" "}
          <strong style={{ color: "var(--ink-2)" }}>{u.refreshedAt}</strong>.
        </span>
        <button className="btn secondary sm" disabled title="Already refreshed today — try tomorrow.">
          Refresh
        </button>
      </div>

      <div className="card" style={{ padding: 22 }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 18 }}>
          <Avatar initials={u.initials} size="xl" tone="amber" />
          <div>
            <div style={{ fontSize: 18, fontWeight: 500 }}>{u.name}</div>
            <div className="muted" style={{ fontSize: 13 }}>
              {u.role}
            </div>
            <div
              className="mono"
              style={{ fontSize: 11, color: "var(--ink-4)", marginTop: 4 }}
            >
              {u.linkedin}
            </div>
          </div>
        </div>
        <hr className="divider" style={{ margin: "8px 0" }} />
        <div className="profile-row">
          <span className="label">Email</span>
          <span>{u.email}</span>
          <span className="muted-2" style={{ fontSize: 11 }}>
            edit
          </span>
        </div>
        <div className="profile-row">
          <span className="label">Education</span>
          <span>{u.education}</span>
          <span className="muted-2" style={{ fontSize: 11 }}>
            edit
          </span>
        </div>
        <div className="profile-row">
          <span className="label">Recent</span>
          <span
            style={{
              minWidth: 0,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {u.recentPosts}
          </span>
          <span className="muted-2" style={{ fontSize: 11 }}>
            edit
          </span>
        </div>
        <div className="profile-row">
          <span className="label">Podcast</span>
          <span>{u.podcasts}</span>
          <span className="muted-2" style={{ fontSize: 11 }}>
            edit
          </span>
        </div>

        <div style={{ marginTop: 22 }}>
          <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>
            Looking for
          </label>
          <textarea
            className="textarea"
            value={u.lookingFor}
            onChange={(e) => setField("lookingFor", e.target.value)}
          />
        </div>
        <div style={{ marginTop: 16 }}>
          <label className="eyebrow" style={{ display: "block", marginBottom: 8 }}>
            Love talking about
          </label>
          <textarea
            className="textarea"
            value={u.talksAbout}
            onChange={(e) => setField("talksAbout", e.target.value)}
          />
        </div>
      </div>
    </div>
  );
}
