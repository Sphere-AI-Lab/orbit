"""Resolve per-arm campaign paths without importing the GPU training stack."""

from collections.abc import Mapping
from pathlib import Path


def resolve_arm_paths(
    repo_root: Path,
    arm_name: str,
    environ: Mapping[str, str],
) -> tuple[Path, Path]:
    """Return the launcher log and checkpoint directory for one arm."""
    log_root = Path(
        environ.get("LORA_REGRET_LOG_DIR", repo_root / "logs" / "lora_regret")
    )
    checkpoint_root = Path(
        environ.get(
            "LORA_REGRET_CKPT_DIR",
            repo_root / "orbit_ckpts" / "lora_regret",
        )
    )
    return log_root / f"{arm_name}.log", checkpoint_root / arm_name
