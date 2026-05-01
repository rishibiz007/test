"use client";
import { useEffect, useState } from "react";
import { AvatarMenu, BrandLink, Toast, type ToastShape } from "@/components/UI";
import Onboarding from "@/components/Onboarding";
import Home from "@/components/Home";
import LoadingScreen from "@/components/LoadingScreen";
import Results from "@/components/Results";
import HistoryPage from "@/components/HistoryPage";
import ProfilePage from "@/components/ProfilePage";
import AdminPage from "@/components/AdminPage";
import { useAppState } from "@/lib/state";
import { MOCK_PEOPLE } from "@/lib/mockPeople";
import type { Person } from "@/lib/types";
import { trackLookupSubmitted, trackLookupResponseReceived, trackLookupFailed } from "@/lib/analytics";

type Route = "onboarding" | "home" | "loading" | "results" | "history" | "profile" | "admin";

export default function Page() {
  const { state, update, reset, hydrated } = useAppState();
  const [route, setRoute] = useState<Route>("home");
  const [loadingHandle, setLoadingHandle] = useState<string | null>(null);
  const [resultPerson, setResultPerson] = useState<Person | null>(null);
  const [lookupDone, setLookupDone] = useState(false);
  const [lookupError, setLookupError] = useState<string | null>(null);
  const [toasts, setToasts] = useState<ToastShape[]>([]);

  useEffect(() => {
    if (!hydrated) return;
    setRoute(state.onboarded ? "home" : "onboarding");
  }, [hydrated, state.onboarded]);

  const pushToast = (t: Omit<ToastShape, "id">) => {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { ...t, id }]);
  };
  const dismissToast = (id: string) => setToasts((prev) => prev.filter((t) => t.id !== id));

  const startLookup = async (handle: string) => {
    trackLookupSubmitted(handle);
    setLoadingHandle(handle);
    setLookupDone(false);
    setLookupError(null);
    setResultPerson(null);
    setRoute("loading");

    try {
      const res = await fetch("/api/lookup", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ handle, user: state.user }),
      });
      const data = (await res.json()) as { person?: Person; error?: string; source?: string };
      if (!res.ok || !data.person) {
        const errMsg = data.error || `Lookup failed (${res.status}).`;
        trackLookupFailed(handle, errMsg);
        setLookupError(errMsg);
        setLookupDone(true);
        return;
      }
      const person = data.person;
      trackLookupResponseReceived({
        target_handle: person.handle,
        target_name: person.name,
        topic_count: person.topics.length,
        source: data.source ?? "live",
      });
      const newCache = { ...state.cache, [person.handle]: person };
      const newHistory = [
        { handle: person.handle, when: new Date().toISOString() },
        ...state.history.filter((h) => h.handle !== person.handle),
      ].slice(0, 50);
      update({
        cache: newCache,
        history: newHistory,
        lastLookup: { handle: person.handle, ts: new Date().toISOString() },
      });
      setResultPerson(person);
      setLookupDone(true);
      if (data.source === "mock") {
        pushToast({
          text: "Demo mode: returning mock topics. Set APIFY_TOKEN + ANTHROPIC_API_KEY for live scraping.",
          ttl: 8000,
        });
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Lookup failed.";
      trackLookupFailed(handle, errMsg);
      setLookupError(errMsg);
      setLookupDone(true);
    }
  };

  const proceedToResults = () => {
    if (resultPerson) {
      setRoute("results");
    }
  };

  const openLookup = (handle: string) => {
    const cached = state.cache[handle] ?? MOCK_PEOPLE[handle];
    if (cached) {
      setResultPerson(cached);
      const newHistory = [
        { handle, when: new Date().toISOString() },
        ...state.history.filter((h) => h.handle !== handle),
      ].slice(0, 50);
      update({
        history: newHistory,
        lastLookup: { handle, ts: new Date().toISOString() },
      });
      setRoute("results");
      return;
    }
    void startLookup(handle);
  };

  const goHome = () => setRoute("home");
  const goHistory = () => setRoute("history");
  const goProfile = () => setRoute("profile");
  const goAdmin = () => setRoute("admin");
  const rerunOnboarding = () => {
    update({ onboarded: false });
    setRoute("onboarding");
  };
  const signOut = () => {
    if (confirm("Sign out and clear your local Spark data?")) {
      reset();
      setRoute("onboarding");
    }
  };

  if (!hydrated) {
    return <div className="app" />;
  }

  const showShell = route !== "onboarding";
  const fallbackPersonForLoading = loadingHandle
    ? state.cache[loadingHandle] ?? MOCK_PEOPLE[loadingHandle] ?? null
    : null;

  return (
    <div className="app">
      {showShell && (
        <header className="topbar">
          <div className="topbar-inner">
            <BrandLink onClick={goHome} />
            <span className="spacer" />
            <AvatarMenu
              user={state.user}
              onProfile={goProfile}
              onRerunOnboarding={rerunOnboarding}
              onAdmin={goAdmin}
              onSignOut={signOut}
            />
          </div>
        </header>
      )}

      {route === "onboarding" && (
        <Onboarding state={state} update={update} onDone={() => setRoute("home")} />
      )}
      {route === "home" && (
        <Home
          state={state}
          onLookup={(h) => void startLookup(h)}
          onOpenHistory={goHistory}
          onOpenLookup={openLookup}
        />
      )}
      {route === "loading" && loadingHandle && (
        <LoadingScreen
          handle={loadingHandle}
          fallbackPerson={fallbackPersonForLoading}
          done={lookupDone && !lookupError}
          error={lookupError}
          onProceed={proceedToResults}
          onCancel={goHome}
        />
      )}
      {route === "results" && resultPerson && (
        <Results
          state={state}
          update={update}
          person={resultPerson}
          onBackHome={goHome}
          pushToast={pushToast}
        />
      )}
      {route === "history" && (
        <HistoryPage state={state} onOpenLookup={openLookup} onBackHome={goHome} />
      )}
      {route === "profile" && <ProfilePage state={state} update={update} />}
      {route === "admin" && <AdminPage onClose={goHome} />}

      <div className="toast-rail">
        {toasts.map((t) => (
          <Toast
            key={t.id}
            text={t.text}
            actionLabel={t.actionLabel}
            onAction={t.onAction}
            onDismiss={() => dismissToast(t.id)}
            ttl={t.ttl ?? 6000}
          />
        ))}
      </div>
    </div>
  );
}
