"""Agentic SWE episode loop (rung 2b) — see generate.py and the design doc
docs/plans/2026-07-07-swe-rung2b-agentic-loop.md."""

from miles.orbit.rewards.swe_agent.container_session import ContainerSession, sif_for_instance
from miles.orbit.rewards.swe_agent.episode import generate

__all__ = ["ContainerSession", "generate", "sif_for_instance"]
