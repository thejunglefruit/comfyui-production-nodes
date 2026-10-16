# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Load this pack inside a real ComfyUI install and smoke-test every node, without starting the server.

    python scripts/check_comfy_load.py /path/to/ComfyUI

Run it with the Python that runs ComfyUI. It imports the pack the way ComfyUI does, calls
`comfy_entrypoint()`, builds every schema (V3 and the V1 view the frontend receives) and
executes each node once on synthetic data. Inputs and outputs go to a temporary folder,
so the ComfyUI install is left untouched.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    comfy_root = Path(sys.argv[1]).resolve()
    sys.argv = [sys.argv[0]]  # ComfyUI parses argv on import
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import folder_paths
    import torch
    from comfy_api.latest import io
    from comfy_execution.graph_utils import ExecutionBlocker

    work = Path(tempfile.mkdtemp(prefix="jf_check_"))
    (work / "input").mkdir()
    (work / "output").mkdir()
    folder_paths.set_input_directory(str(work / "input"))
    folder_paths.set_output_directory(str(work / "output"))

    spec = importlib.util.spec_from_file_location(
        "comfyui_production_nodes", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    extension = asyncio.run(module.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    print(f"comfy_entrypoint() -> {type(extension).__name__} with {len(nodes)} nodes")
    by_id = {}
    for node in nodes:
        schema = node.define_schema()
        v1 = node.GET_NODE_INFO_V1()
        node.INPUT_TYPES()
        by_id[schema.node_id] = node
        print(f"  OK {schema.node_id:28s} '{schema.display_name}'  [{schema.category}]  "
              f"inputs={len(schema.inputs)} outputs={len(schema.outputs)} output_node={v1['output_node']}")

    # 0. The API example must pass ComfyUI's own prompt validation (only a missing checkpoint is tolerated)
    import execution
    import nodes as comfy_nodes
    comfy_nodes.NODE_CLASS_MAPPINGS.update(by_id)  # in-process only
    example = json.loads((PACK / "examples" / "production_batch_api.json").read_text(encoding="utf-8"))
    valid, error, good_outputs, node_errors = asyncio.run(execution.validate_prompt("check", example, None))
    problems = [(nid, e) for nid, info in node_errors.items() for e in info["errors"]
                if not (e["type"] == "value_not_in_list" and e["extra_info"]["input_name"] == "ckpt_name")]
    assert not problems, problems
    tolerated = sum(len(info["errors"]) for info in node_errors.values())
    print(f"  OK example API workflow validates ({len(example)} nodes; tolerated missing-checkpoint errors: {tolerated})")

    def run(node_id, hidden=None, **kwargs):
        clone = by_id[node_id].PREPARE_CLASS_CLONE({"hidden_inputs": hidden or {}})
        return clone.EXECUTE_NORMALIZED(**kwargs)

    # 1. Prompt Batch from CSV, from a file in the (temporary) input folder
    (work / "input" / "prompts.csv").write_text("prompt,negative,seed,steps,tag\nfox,blurry,42,30,a\nowl,,7,,b\n", encoding="utf-8")
    out = run("JungleFruit_PromptBatchCSV", csv_path="prompts.csv", csv_text="", index=3, on_overflow="wrap")
    assert out.result[0] == "owl" and out.result[2] == 7 and out.result[8] == 1 and out.result[9] == 2, out.result
    try:
        run("JungleFruit_PromptBatchCSV", csv_path="../outside.csv", csv_text="", index=0, on_overflow="wrap")
        raise AssertionError("path traversal was not rejected")
    except ValueError as exc:
        print(f"  OK PromptBatchCSV  row=1 prompt='owl' seed=7; traversal rejected: {exc}")

    # 2. Quality Gate: one sharp checkerboard, one black frame
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    board = ((((x // 4) + (y // 4)) % 2).float() * 0.5 + 0.25).unsqueeze(-1).expand(64, 64, 3)
    batch = torch.stack([board, torch.zeros(64, 64, 3)])
    out = run("JungleFruit_QualityGate", images=batch, min_sharpness=50.0, min_brightness=0.08, max_brightness=0.92,
              max_clipped_highlights=0.05, max_crushed_shadows=0.05, min_contrast=0.04, if_none_pass="return_best")
    passed, rejected, report, count = out.result
    assert passed.shape[0] == 1 and rejected.shape[0] == 1 and count == 1
    out = run("JungleFruit_QualityGate", images=batch[:1], min_sharpness=50.0, min_brightness=0.08, max_brightness=0.92,
              max_clipped_highlights=0.05, max_crushed_shadows=0.05, min_contrast=0.04, if_none_pass="block")
    assert isinstance(out.result[1], ExecutionBlocker)
    print(f"  OK QualityGate     passed={passed.shape[0]} rejected={rejected.shape[0]}; empty branch -> ExecutionBlocker")

    # 3. Apply .cube LUT with a bundled look
    luts = [n for n in folder_paths.get_filename_list("luts") if n.endswith(".cube")]
    assert "teal_orange.cube" in luts, luts
    out = run("JungleFruit_ApplyCubeLUT", image=batch, lut_name="teal_orange.cube", intensity=0.8, lut_path="")
    assert out.result[0].shape == batch.shape
    print(f"  OK ApplyCubeLUT    luts folder lists {luts}")

    # 4. Run Manifest: saves PNGs + sidecars into the temporary output folder
    graph = {"1": {"class_type": "JungleFruit_PromptBatchCSV", "inputs": {"index": 3}}}
    out = run("JungleFruit_RunManifest", hidden={io.Hidden.prompt: graph, io.Hidden.extra_pnginfo: {"workflow": {}}},
              images=batch, filename_prefix="check/run", positive="owl", negative="", seed=7, tag="b",
              params="cfg=6.5", include_graph=True, include_workflow=False)
    files = sorted(p.name for p in (work / "output" / "check").iterdir())
    assert files == ["run_00001_.json", "run_00001_.png", "run_00002_.json", "run_00002_.png"], files
    sidecar = json.loads((work / "output" / "check" / "run_00001_.json").read_text(encoding="utf-8"))
    assert sidecar["provenance"]["prompt_sha256"] and sidecar["outputs"][0]["file"] == "run_00001_.png"
    print(f"  OK RunManifest     wrote {files}")
    print(f"     prompt_sha256={sidecar['provenance']['prompt_sha256'][:16]}...  params={sidecar['generation']['params']}")
    shutil.rmtree(work, ignore_errors=True)
    print("ALL NODES LOADED AND EXECUTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
