# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
import datetime as dt
import json

import pytest

from jf_nodes.core.manifest import (
    SCHEMA_ID, build_manifest, graph_skeleton, parse_params, sha256_of, sidecar_path, write_json_atomic,
)


def api_graph(seed=42, text="a red fox"):
    return {
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20, "model": ["4", 0], "positive": ["6", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["4", 1]}},
    }


NOW = dt.datetime(2026, 1, 2, 3, 4, 5, 999, tzinfo=dt.timezone.utc)


def test_full_hash_changes_with_values_skeleton_hash_does_not():
    a, b = api_graph(seed=1), api_graph(seed=2, text="other")
    ma, mb = build_manifest(prompt_graph=a, now=NOW), build_manifest(prompt_graph=b, now=NOW)
    assert ma["provenance"]["prompt_sha256"] != mb["provenance"]["prompt_sha256"]
    assert ma["provenance"]["graph_sha256"] == mb["provenance"]["graph_sha256"]


def test_hash_is_independent_of_key_order():
    g = api_graph()
    reordered = {k: g[k] for k in reversed(list(g))}
    assert sha256_of(g) == sha256_of(reordered)


def test_skeleton_keeps_links_and_types_only():
    sk = graph_skeleton(api_graph())
    assert sk["3"]["inputs"]["model"] == ["link", "4", 0]
    assert sk["3"]["inputs"]["seed"] == "int"
    assert sk["6"]["inputs"]["text"] == "str"


def test_manifest_contents_and_determinism():
    kwargs = dict(positive="a red fox", negative="blurry", seed=42, tag="fox", params={"cfg": 6.5},
                  outputs=[{"file": "run_00001_.png", "subfolder": "", "batch_index": 0}],
                  prompt_graph=api_graph(), tool={"name": "t", "version": "1"}, now=NOW)
    m = build_manifest(**kwargs)
    assert m == build_manifest(**kwargs)
    assert m["schema"] == SCHEMA_ID
    assert m["created_utc"] == "2026-01-02T03:04:05Z"
    assert m["generation"] == {"positive": "a red fox", "negative": "blurry", "seed": 42, "tag": "fox", "params": {"cfg": 6.5}}
    assert m["provenance"]["node_count"] == 3
    assert m["provenance"]["class_types"] == ["CLIPTextEncode", "CheckpointLoaderSimple", "KSampler"]
    assert m["graph"] == api_graph() and "workflow" not in m


def test_graph_and_workflow_switches():
    m = build_manifest(prompt_graph=api_graph(), workflow={"nodes": []}, include_graph=False, include_workflow=True)
    assert "graph" not in m and m["workflow"] == {"nodes": []}
    assert m["provenance"]["prompt_sha256"]  # hash is kept even when the graph itself is not


def test_manifest_without_graph_is_still_valid():
    m = build_manifest(seed=1)
    assert m["provenance"] == {"prompt_sha256": None, "graph_sha256": None, "node_count": 0, "class_types": []}


def test_no_environment_data_leaks():
    text = json.dumps(build_manifest(prompt_graph=api_graph(), now=NOW))
    for key in ("hostname", "user", "home", "cwd", "platform"):
        assert f'"{key}"' not in text


def test_parse_params_formats():
    assert parse_params("") == {}
    assert parse_params('{"cfg": 6.5, "lut": "teal"}') == {"cfg": 6.5, "lut": "teal"}
    assert parse_params("cfg=6.5\nsteps = 28\n# note\nhires=true\nlut=teal_orange") == {
        "cfg": 6.5, "steps": 28, "hires": True, "lut": "teal_orange"}
    with pytest.raises(ValueError, match="key=value"):
        parse_params("just words")
    with pytest.raises(ValueError, match="object"):
        parse_params("[1, 2]")


def test_sidecar_written_atomically(tmp_path):
    image = tmp_path / "run_00001_.png"
    target = sidecar_path(str(image))
    assert target.endswith("run_00001_.json")
    write_json_atomic(target, {"a": 1})
    assert json.loads(open(target, encoding="utf-8").read()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["run_00001_.json"]  # no temp file left behind
