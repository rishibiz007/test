"use client";
import { useMemo, useState } from "react";
import { Avatar, Icon } from "./UI";
import { MOCK_PEOPLE } from "@/lib/mockPeople";
import { relativeTime } from "@/lib/state";
import type { AppState } from "@/lib/types";

interface Props {
  state: AppState;
  onOpenLookup: (handle: string) => void;
  onBackHome: () => void;
}

export default function HistoryPage({ state, onOpenLookup, onBackHome }: Props) {
  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const items = state.history
      .map((it) => ({ ...it, person: state.cache[it.handle] ?? MOCK_PEOPLE[it.handle] }))
      .filter((x) => x.person);
    if (!q.trim()) return items;
    const t = q.toLowerCase();
    return items.filter(
      (x) =>
        x.person.name.toLowerCase().includes(t) ||
        x.person.company.toLowerCase().includes(t) ||
        x.handle.toLowerCase().includes(t)
    );
  }, [q, state.history, state.cache]);

  return (
    <div className="page narrow fade-in" data-screen-label="04 history">
      <div style={{ marginBottom: 22 }}>
        <span className="eyebrow">HISTORY</span>
      </div>
      <h1 className="h-1 serif" style={{ marginBottom: 22 }}>
        Your lookups
      </h1>
      <div className="search-wrap" style={{ marginBottom: 22 }}>
        <Icon name="search" size={15} />
        <input
          placeholder="Search by name, company, or URL…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          autoFocus
        />
      </div>
      {filtered.length === 0 ? (
        <div className="empty">
          <div className="em-title">{q ? "No matches" : "Nothing here yet."}</div>
          <div style={{ fontSize: 13 }}>
            {q ? "Try a different search term." : "Look up someone to start filling this list."}
          </div>
          {!q && (
            <button className="btn secondary sm" style={{ marginTop: 14 }} onClick={onBackHome}>
              Look someone up →
            </button>
          )}
        </div>
      ) : (
        <div className="card">
          {filtered.map((it, i) => (
            <a
              key={i}
              href="#"
              className="history-row"
              onClick={(e) => {
                e.preventDefault();
                onOpenLookup(it.handle);
              }}
            >
              <Avatar initials={it.person.initials} size="md" />
              <div style={{ minWidth: 0 }}>
                <div className="person-name">{it.person.name}</div>
                <div className="person-co">
                  {it.person.company} · {it.person.role}
                </div>
              </div>
              <span
                className="muted-2"
                style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
              >
                {relativeTime(it.when)}
              </span>
              <span className="badge success">cached</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
