"""Arize AX tracing setup for the Phoenix PM agent.

Call `setup_tracing()` once at process start, before any Anthropic client is
created. Returns the TracerProvider so the caller can force_flush + shutdown
before exit (required for short-lived CLI scripts — otherwise OTLP exports
are dropped).
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def setup_tracing(project_name: str = "phoenix-pm-agent") -> Optional[object]:
    """Register Arize AX exporter + Anthropic auto-instrumentor.

    Fails gracefully (warns + returns None) if ARIZE_API_KEY / ARIZE_SPACE_ID
    are missing — the agent still runs, just without tracing.
    """
    api_key = os.environ.get("ARIZE_API_KEY")
    space_id = os.environ.get("ARIZE_SPACE_ID") or os.environ.get("ARIZE_SPACE")

    if not api_key or not space_id:
        print(
            "[instrumentation] ARIZE_API_KEY / ARIZE_SPACE_ID not set — "
            "skipping tracing. Set both env vars to enable Arize AX traces.",
            file=sys.stderr,
        )
        return None

    from arize.otel import register
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    tracer_provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name=project_name,
    )
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    print(
        f"[instrumentation] Arize AX tracing active — project={project_name!r}",
        file=sys.stderr,
    )
    return tracer_provider


def shutdown_tracing(tracer_provider: Optional[object]) -> None:
    """Flush pending spans and shut down the exporter.

    Required for short-lived scripts: OTLP export is async; without flushing
    before exit, the last spans (often the most important ones) are lost.
    """
    if tracer_provider is None:
        return
    try:
        tracer_provider.force_flush()  # type: ignore[attr-defined]
        tracer_provider.shutdown()      # type: ignore[attr-defined]
    except Exception as e:
        print(f"[instrumentation] flush/shutdown error: {e}", file=sys.stderr)
