"""Orbit-facing Megatron Bridge package boundary."""

# ORBIT-SEAM: single import boundary for megatron.bridge; orbit code imports AutoBridge from here
from megatron.bridge import AutoBridge


__all__ = ["AutoBridge"]
