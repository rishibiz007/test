"use client";
import { useEffect } from "react";
import { initAmplitude } from "@/lib/analytics";

export default function AmplitudeInit() {
  useEffect(() => { initAmplitude(); }, []);
  return null;
}
