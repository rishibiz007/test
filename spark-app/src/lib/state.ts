"use client";
import { useEffect, useState } from "react";
import type { AppState } from "./types";
import { DEFAULT_USER } from "./mockPeople";

const STORAGE_KEY = "spark.app.v1";

export const initialState = (): AppState => ({
  onboarded: false,
  user: { ...DEFAULT_USER },
  ratings: {},
  history: [],
  lastLookup: null,
  cache: {},
});

function loadState(): AppState {
  if (typeof window === "undefined") return initialState();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return initialState();
    const parsed = JSON.parse(raw);
    return { ...initialState(), ...parsed };
  } catch {
    return initialState();
  }
}

function saveState(s: AppState) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {}
}

export function useAppState() {
  const [state, setStateInternal] = useState<AppState>(initialState);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setStateInternal(loadState());
    setHydrated(true);
  }, []);

  const update = (patch: Partial<AppState>) => {
    setStateInternal((prev) => {
      const next = { ...prev, ...patch };
      saveState(next);
      return next;
    });
  };

  const reset = () => {
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(STORAGE_KEY);
    }
    setStateInternal(initialState());
  };

  return { state, update, reset, hydrated };
}

export function relativeTime(iso: string): string {
  const d = new Date(iso);
  const diffMs = Date.now() - d.getTime();
  const m = Math.round(diffMs / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const dd = Math.round(h / 24);
  if (dd === 1) return "yesterday";
  if (dd < 7) return `${dd}d ago`;
  const w = Math.round(dd / 7);
  return `${w}w ago`;
}
