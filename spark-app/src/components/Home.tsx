"use client";
import { useEffect, useRef, useState } from "react";
import { Avatar, Icon } from "./UI";
import { MOCK_PEOPLE } from "@/lib/mockPeople";
import { relativeTime } from "@/lib/state";
import type { AppState } from "@/lib/types";

interface Props {
  state: AppState;
  onLookup: (handle: string) => void;
  onOpenHistory: () => void;
  onOpenLookup: (handle: string) => void;
}

export default function Home({ state, onLookup, onOpenHistory, onOpenLookup }: Props) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const submit = () => {
    const cleaned = url.trim().replace(/^https?:\/\//, "").replace(/^www\./, "").replace(/\/$/, "");
    if (!cleaned) return;
    if (!cleaned.includes("linkedin.com/in/")) {
      setError("That doesn't look like a LinkedIn profile URL.");
      return;
    }
    setError("");
    setUrl("");
    onLookup(cleaned);
  };

  const recent = state.history.slice(0, 4);
  const suggestions = Object.values(MOCK_PEOPLE).map((p) => p.handle);

  return (
    <div className="page narrow fade-in" data-screen-label="02 home">
      <div style={{ marginBottom: 32 }}>
        <span className="eyebrow">NEW LOOKUP</span>
      </div>
      <h1 className="h-display serif" style={{ marginBottom: 14 }}>
        Who are you meeting?
      </h1>
      <p className="muted" style={{ marginBottom: 28, fontSize: 15, maxWidth: 520 }}>
        Paste a LinkedIn URL. Ice Breaker reads their profile, recent posts, and any podcast or blog
        appearances — then surfaces what you actually have in common.
      </p>

      <div style={{ position: "relative" }}>
        <div className="search-wrap" style={{ height: 44, padding: "0 8px 0 14px" }}>
          <Icon name="search" size={15} />
          <input
            ref={inputRef}
            placeholder="linkedin.com/in/…"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              setError("");
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") submit();
            }}
            className="mono"
            style={{ fontSize: 13 }}
          />
          <span className="kbd" style={{ marginRight: 8 }}>
            ⌘ K
          </span>
          <button className="btn" onClick={submit} disabled={!url.trim()}>
            Look up <Icon name="arrow" />
          </button>
        </div>
        {error && (
          <div
            style={{
              marginTop: 10,
              fontSize: 12,
              color: "var(--bad)",
              background: "var(--bad-soft)",
              border: "1px solid #f0c8c4",
              padding: "8px 12px",
              borderRadius: "var(--r-1)",
            }}
          >
            {error}
          </div>
        )}
        <div
          className="muted-2"
          style={{
            fontSize: 11,
            marginTop: 10,
            fontFamily: "JetBrains Mono, monospace",
          }}
        >
          public info only · personalized using your profile · cached 24h
        </div>
      </div>

      <div style={{ marginTop: 28 }}>
        <div className="eyebrow" style={{ marginBottom: 10 }}>
          TRY ONE OF THESE
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {suggestions.map((h) => {
            const p = MOCK_PEOPLE[h];
            return (
              <button
                key={h}
                className="chip"
                onClick={() => {
                  setUrl(h);
                  setError("");
                  setTimeout(() => onLookup(h), 60);
                }}
              >
                {p.name} · {p.company}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ marginTop: 48 }}>
        <div className="sec-head">
          <span className="eyebrow">RECENT</span>
          {state.history.length > 4 && (
            <a
              href="#"
              className="link"
              onClick={(e) => {
                e.preventDefault();
                onOpenHistory();
              }}
            >
              See all →
            </a>
          )}
        </div>
        {recent.length === 0 ? (
          <div className="empty">
            <div className="em-title">No lookups yet.</div>
            <div style={{ fontSize: 13 }}>Try one of the suggested profiles above to get started.</div>
          </div>
        ) : (
          <div className="card stagger">
            {recent.map((it, i) => {
              const cached = state.cache[it.handle] ?? MOCK_PEOPLE[it.handle];
              if (!cached) return null;
              return (
                <a
                  key={i}
                  href="#"
                  className="history-row"
                  onClick={(e) => {
                    e.preventDefault();
                    onOpenLookup(it.handle);
                  }}
                >
                  <Avatar initials={cached.initials} size="md" />
                  <div style={{ minWidth: 0 }}>
                    <div className="person-name">{cached.name}</div>
                    <div className="person-co">{cached.company}</div>
                  </div>
                  <span
                    className="muted-2"
                    style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
                  >
                    {relativeTime(it.when)}
                  </span>
                  <span className="badge success" title="Cached snapshot — no re-scrape">
                    cached
                  </span>
                </a>
              );
            })}
            <a
              href="#"
              className="history-row"
              onClick={(e) => {
                e.preventDefault();
                onOpenHistory();
              }}
              style={{ color: "var(--ink-3)", borderTop: "1px solid var(--line-2)" }}
            >
              <span style={{ width: 36, textAlign: "center", color: "var(--ink-4)" }}>↗</span>
              <div className="person-name" style={{ color: "var(--ink-2)", fontSize: 13 }}>
                See all lookups
              </div>
              <span />
              <span />
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
