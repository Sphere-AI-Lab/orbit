"""Declarative OPD teacher pools: several named frozen teachers, mixed transports.

Dev-native replacement for the ultra tier's manifest/pool pair (whose identity
chain is welded to the qualification preflight). A pool manifest declares named
teachers of two kinds:

- ``url``:    an externally served endpoint, used as-is.
- ``served``: a HF checkpoint this job serves itself -- one extra sglang model
              entry per teacher, exactly like ``--opd-serve-teacher`` (same
              scoring-correctness server flags), on GPUs after the rollout
              bucket.

After the served engines start, every teacher resolves to a routing entry
``name=url@weight`` and the pool is handed to the EXISTING multi-teacher
router (``--opd-teacher-urls`` semantics: per-sample routing via
``sample.metadata[--opd-teacher-key]``, weighted ensembles per name, the
reserved name ``default`` as fallback).

Manifest (YAML or JSON)::

    teachers:
      - name: math          # routing name; one entry may be named "default"
        kind: served
        model_path: /ckpts/Qwen2.5-7B-Instruct
        num_gpus: 1         # default 1; one engine, TP across them
        mem_fraction: 0.6   # optional mem_fraction_static override
        weight: 1.0         # optional mixture weight (ensembles share a name)
      - name: default
        kind: url
        url: http://host:30001/generate

Sampled-token scoring only: full-vocab reconstruction needs one LM head per
member trainer-side, which stays single-teacher for now.
"""

from __future__ import annotations

import dataclasses
import json
import os

import yaml

_VALID_KINDS = ("url", "served")


class TeacherPoolError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class TeacherPoolEntry:
    name: str
    kind: str
    url: str | None = None
    model_path: str | None = None
    num_gpus: int = 1
    num_gpus_per_engine: int | None = None
    mem_fraction: float | None = None
    weight: float = 1.0

    @property
    def served_model_name(self) -> str:
        return f"opd_teacher_{self.name}"


@dataclasses.dataclass(frozen=True)
class TeacherPool:
    entries: tuple[TeacherPoolEntry, ...]

    @property
    def served(self) -> tuple[TeacherPoolEntry, ...]:
        return tuple(e for e in self.entries if e.kind == "served")

    @property
    def served_num_gpus(self) -> int:
        return sum(e.num_gpus for e in self.served)

    def routing_specs(self, served_urls: dict[str, str]) -> list[str]:
        """``name=url@weight`` strings for parse_teacher_urls, once served
        entries' router URLs are known (keyed by served_model_name)."""
        specs: dict[str, list[str]] = {}
        for entry in self.entries:
            if entry.kind == "url":
                url = entry.url
            else:
                url = served_urls.get(entry.served_model_name)
                if url is None:
                    raise TeacherPoolError(
                        f"served teacher {entry.name!r} has no published endpoint; "
                        "was its engine started?"
                    )
            specs.setdefault(entry.name, []).append(f"{url}@{entry.weight}")
        return [f"{name}={','.join(urls)}" for name, urls in specs.items()]


def _exact_str(value: object, *, field: str, index: int) -> str:
    if type(value) is not str or not value.strip():
        raise TeacherPoolError(f"teacher[{index}].{field} must be a non-empty string")
    return value


def parse_teacher_pool(path: str) -> TeacherPool:
    if not os.path.isfile(path):
        raise TeacherPoolError(f"teacher pool manifest not found: {path}")
    with open(path) as handle:
        raw = handle.read()
    data = json.loads(raw) if path.endswith(".json") else yaml.safe_load(raw)
    if type(data) is not dict or set(data) != {"teachers"}:
        raise TeacherPoolError("teacher pool manifest must be an object with exactly a 'teachers' list")
    items = data["teachers"]
    if type(items) is not list or not items:
        raise TeacherPoolError("teacher pool manifest 'teachers' must be a non-empty list")

    entries: list[TeacherPoolEntry] = []
    for i, item in enumerate(items):
        if type(item) is not dict:
            raise TeacherPoolError(f"teacher[{i}] must be an object")
        unknown = set(item) - {"name", "kind", "url", "model_path", "num_gpus", "num_gpus_per_engine", "mem_fraction", "weight"}
        if unknown:
            raise TeacherPoolError(f"teacher[{i}] has unknown fields: {sorted(unknown)}")
        name = _exact_str(item.get("name"), field="name", index=i)
        kind = _exact_str(item.get("kind"), field="kind", index=i)
        if kind not in _VALID_KINDS:
            raise TeacherPoolError(f"teacher[{i}].kind must be one of {_VALID_KINDS}, got {kind!r}")
        weight = item.get("weight", 1.0)
        if type(weight) not in (int, float) or not weight > 0:
            raise TeacherPoolError(f"teacher[{i}].weight must be a positive number")
        if kind == "url":
            if "model_path" in item or "num_gpus" in item or "mem_fraction" in item or "num_gpus_per_engine" in item:
                raise TeacherPoolError(f"teacher[{i}] kind=url takes only name/kind/url/weight")
            entries.append(
                TeacherPoolEntry(name=name, kind=kind, url=_exact_str(item.get("url"), field="url", index=i), weight=float(weight))
            )
            continue
        model_path = _exact_str(item.get("model_path"), field="model_path", index=i)
        if "url" in item:
            raise TeacherPoolError(f"teacher[{i}] kind=served does not take a url")
        num_gpus = item.get("num_gpus", 1)
        if type(num_gpus) is not int or num_gpus < 1:
            raise TeacherPoolError(f"teacher[{i}].num_gpus must be a positive integer")
        num_gpus_per_engine = item.get("num_gpus_per_engine")
        if num_gpus_per_engine is not None and (type(num_gpus_per_engine) is not int or num_gpus_per_engine < 1):
            raise TeacherPoolError(f"teacher[{i}].num_gpus_per_engine must be a positive integer")
        mem_fraction = item.get("mem_fraction")
        if mem_fraction is not None and (type(mem_fraction) not in (int, float) or not 0 < mem_fraction <= 1):
            raise TeacherPoolError(f"teacher[{i}].mem_fraction must be in (0, 1]")
        entries.append(
            TeacherPoolEntry(
                name=name,
                kind=kind,
                model_path=model_path,
                num_gpus=num_gpus,
                num_gpus_per_engine=num_gpus_per_engine,
                mem_fraction=float(mem_fraction) if mem_fraction is not None else None,
                weight=float(weight),
            )
        )

    served_names = [e.served_model_name for e in entries if e.kind == "served"]
    if len(served_names) != len(set(served_names)):
        raise TeacherPoolError("served teacher names must be unique (they become sglang model entries)")
    return TeacherPool(entries=tuple(entries))
