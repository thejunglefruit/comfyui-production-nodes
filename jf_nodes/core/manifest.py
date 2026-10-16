# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Provenance sidecars: a small JSON file next to every rendered image.

A manifest answers "how exactly was this frame made?" months later, without opening
ComfyUI: prompt, seed, sampler parameters, a hash of the exact API graph that ran, and a
structure-only hash that groups runs of the same workflow regardless of widget values.

Privacy by design: no host name, user name, absolute path or environment data is
recorded. Only what is passed in (and, optionally, the graph itself) ends up in the file.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
from typing import Any

SCHEMA_ID = "jungle-fruit/run-manifest@1"


def canonical_json(obj: Any) -> str:
    """Stable JSON text: sorted keys, no whitespace, non-serialisable values stringified."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def graph_skeleton(prompt_graph: dict | None) -> dict:
    """Reduce an API-format graph to its structure: node types, links and input *types*.

    Two runs of the same workflow with different seeds or prompts share a skeleton,
    so its hash identifies "which workflow" while the full hash identifies "which run".
    """
    if not prompt_graph:
        return {}
    skeleton = {}
    for node_id, node in sorted(prompt_graph.items(), key=lambda kv: str(kv[0])):
        if not isinstance(node, dict):
            continue
        inputs = {}
        for name, value in sorted((node.get("inputs") or {}).items()):
            inputs[name] = ["link", value[0], value[1]] if _is_link(value) else type(value).__name__
        skeleton[str(node_id)] = {"class_type": node.get("class_type"), "inputs": inputs}
    return skeleton


def parse_params(text: str) -> dict:
    """Accept a JSON object or `key=value` lines (numbers and booleans are converted)."""
    text = (text or "").strip()
    if not text:
        return {}
    if text[0] in "{[":
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("params JSON must be an object")
        return value
    out: dict[str, Any] = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"params line {n}: expected key=value, got '{line}'")
        key, raw = (part.strip() for part in line.split("=", 1))
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def utc_timestamp(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_manifest(
    *,
    positive: str = "",
    negative: str = "",
    seed: int | None = None,
    tag: str = "",
    params: dict | None = None,
    outputs: list[dict] | None = None,
    prompt_graph: dict | None = None,
    workflow: dict | None = None,
    include_graph: bool = True,
    include_workflow: bool = False,
    tool: dict | None = None,
    now: _dt.datetime | None = None,
) -> dict:
    """Assemble the manifest dictionary. Pure: same inputs (and `now`) give the same output."""
    skeleton = graph_skeleton(prompt_graph)
    class_types = sorted({n.get("class_type") for n in skeleton.values() if n.get("class_type")})
    manifest: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "created_utc": utc_timestamp(now),
        "tool": tool or {},
        "generation": {
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "tag": tag,
            "params": params or {},
        },
        "outputs": outputs or [],
        "provenance": {
            "prompt_sha256": sha256_of(prompt_graph) if prompt_graph else None,
            "graph_sha256": sha256_of(skeleton) if skeleton else None,
            "node_count": len(skeleton),
            "class_types": class_types,
        },
    }
    if include_graph and prompt_graph:
        manifest["graph"] = prompt_graph
    if include_workflow and workflow:
        manifest["workflow"] = workflow
    return manifest


def write_json_atomic(path: str, data: dict) -> None:
    """Write JSON via a temp file + rename so a crash never leaves a half-written sidecar."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def sidecar_path(image_path: str) -> str:
    """`render_00001_.png` -> `render_00001_.json` (same folder, same stem)."""
    return os.path.splitext(image_path)[0] + ".json"
