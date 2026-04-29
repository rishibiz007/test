"use client";
import { useEffect, useState } from "react";
import { Avatar } from "./UI";
import type { Person } from "@/lib/types";

interface Props {
  handle: string;
  fallbackPerson?: Person | null;
  done: boolean;
  error: string | null;
  onProceed: () => void;
  onCancel: () => void;
}

export default function LoadingScreen({
  handle,
  fallbackPerson,
  done,
  error,
  onProceed,
  onCancel,
}: Props) {
  const steps = [
    "Triggering Apify LinkedIn actor",
    "Reading profile + recent posts",
    "Synthesizing topics with Claude",
    "Finding overlap with your profile",
  ];
  const [active, setActive] = useState(0);

  useEffect(() => {
    const timers: ReturnType<typeof setTimeout>[] = [];
    [620, 1380, 2200, 3100].forEach((ms, i) => {
      timers.push(setTimeout(() => setActive(i + 1), ms));
    });
    return () => timers.forEach(clearTimeout);
  }, []);

  useEffect(() => {
    if (done && active >= 4) onProceed();
  }, [done, active, onProceed]);

  const initials =
    fallbackPerson?.initials ||
    handle
      .replace(/.*\/in\//, "")
      .slice(0, 2)
      .toUpperCase() ||
    "??";

  return (
    <div className="loading fade-in" data-screen-label="03 loading">
      <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 28 }}>
        <Avatar initials={initials} size="lg" />
        <div>
          <div className="serif" style={{ fontSize: 22, lineHeight: 1.1 }}>
            {fallbackPerson?.name || handle.replace(/.*\/in\//, "")}
          </div>
          <div className="muted" style={{ fontSize: 13 }}>
            {fallbackPerson?.role
              ? `${fallbackPerson.role} · ${fallbackPerson.company}`
              : handle}
          </div>
        </div>
      </div>
      <div>
        {steps.map((s, i) => (
          <div
            key={s}
            className={`progress-line ${i < active ? "done" : i === active ? "active" : "pending"}`}
          >
            <span className="progress-dot" />
            <span>
              {i < active ? "✓ " : i === active ? "→ " : "  "}
              {s}
              {i === active && "…"}
            </span>
          </div>
        ))}
      </div>
      {error && (
        <div
          style={{
            marginTop: 24,
            padding: "12px 14px",
            background: "var(--bad-soft)",
            border: "1px solid #f0c8c4",
            color: "#8a3933",
            fontSize: 13,
            borderRadius: "var(--r-1)",
          }}
        >
          {error}
          <div style={{ marginTop: 10 }}>
            <button className="btn secondary sm" onClick={onCancel}>
              ← Back
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
