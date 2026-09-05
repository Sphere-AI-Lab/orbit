from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_extract_pins() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "slurm" / "setup" / "extract_pins.py"
    spec = importlib.util.spec_from_file_location("orbit_extract_pins_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


extract_pins = _load_extract_pins()


def _pin_values(
    *,
    source_version: str = "v0.5.15",
    upstream_image_tag: str = "v0.5.15",
    upstream_wheels_tag: str = "cu130-x86_64",
) -> dict[str, str]:
    return {
        "ORBIT_SGLANG_SOURCE_VERSION": source_version,
        "MILES_WHEELS_TAG": "cu129-x86_64",
        "MILES_WHEELS_TORCH_VERSION": "2.11.0",
        "MILES_WHEELS_SGLANG_VERSION": "v0.5.15",
        "UPSTREAM_SGLANG_IMAGE_TAG": upstream_image_tag,
        "UPSTREAM_WHEELS_TAG": upstream_wheels_tag,
    }


def _render_values() -> dict[str, str]:
    values = {pin.key: f"test-{pin.key.lower()}" for _, pins in extract_pins.PIN_GROUPS for pin in pins}
    values.update(_pin_values())
    values.update(
        {
            "SGLANG_ROUTER_VERSION": "0.3.2",
            "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu129",
            "FLASHINFER_INDEX_URL": "https://flashinfer.ai/whl/cu129",
            "SGL_WHL_INDEX_URL": "https://docs.sglang.ai/whl/cu129",
            # the hand-owned pin (FA3 interface since the 2026-08 sync; TMS is extracted again)
            "FLASH_ATTN_INTERFACE_COMMIT": "deadbeef",
        }
    )
    return values


def test_versionless_upstream_wheels_allow_bundle_to_lag_matching_source() -> None:
    assert extract_pins.pending_notice(_pin_values()) is None


def test_versionless_upstream_wheels_report_source_lag() -> None:
    notice = extract_pins.pending_notice(_pin_values(source_version="v0.5.14"))

    assert notice is not None
    assert "ORBIT_SGLANG_SOURCE_VERSION=v0.5.14" in notice
    assert "UPSTREAM_SGLANG_IMAGE_TAG=v0.5.15" in notice


def test_versioned_upstream_wheels_still_compare_source_to_image() -> None:
    notice = extract_pins.pending_notice(
        _pin_values(
            source_version="v0.5.14",
            upstream_wheels_tag="cu130-x86_64-v0.5.15",
        )
    )

    assert notice is not None


def test_render_records_hand_owned_source_version() -> None:
    assert "ORBIT_SGLANG_SOURCE_VERSION=${ORBIT_SGLANG_SOURCE_VERSION:-v0.5.15}" in extract_pins.render(
        _render_values()
    )
