# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""JungleFruit_RunManifest: save images plus a provenance JSON sidecar for each one."""

from __future__ import annotations

import json
import os

import folder_paths
from comfy_api.latest import io, ui

from .. import __version__
from ..core.manifest import build_manifest, parse_params, sidecar_path, write_json_atomic

TOOL = {"name": "comfyui-production-nodes", "version": __version__}


class RunManifest(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JungleFruit_RunManifest",
            display_name="Save Images + Run Manifest",
            category="jungle_fruit/output",
            description=(
                "Saves images as PNG and writes a JSON sidecar next to each one with prompt, seed, parameters, "
                "a SHA-256 of the exact graph that ran and a timestamp. No host, user or path data is recorded."
            ),
            search_aliases=["provenance", "metadata", "sidecar", "save with json"],
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="jf/run",
                                tooltip="Relative to the output folder; subfolders allowed (e.g. client/shot010)."),
                io.String.Input("positive", default="", multiline=True, optional=True),
                io.String.Input("negative", default="", multiline=True, optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, optional=True),
                io.String.Input("tag", default="", optional=True),
                io.String.Input("params", default="", multiline=True, optional=True,
                                placeholder="JSON object or key=value lines, e.g.\ncfg=6.5\nsteps=28\nlut=teal_orange",
                                tooltip="Free-form parameters to record (sampler, cfg, LUT, client, shot...)."),
                io.Boolean.Input("include_graph", default=True,
                                 tooltip="Embed the API-format graph so the run can be re-queued from the sidecar."),
                io.Boolean.Input("include_workflow", default=False, advanced=True,
                                 tooltip="Also embed the UI workflow (larger; contains node positions and titles)."),
            ],
            outputs=[io.String.Output("manifest", tooltip="The manifest of the first image, as JSON text.")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, images, filename_prefix, positive="", negative="", seed=0, tag="", params="",
                include_graph=True, include_workflow=False) -> io.NodeOutput:
        try:
            parsed_params = parse_params(params)
        except ValueError as exc:
            raise ValueError(f"[Run Manifest] params: {exc}") from None

        saved = ui.ImageSaveHelper.save_images(images, filename_prefix, io.FolderType.output, cls)
        output_dir = folder_paths.get_output_directory()
        workflow = (cls.hidden.extra_pnginfo or {}).get("workflow") if include_workflow else None

        first = None
        for batch_index, result in enumerate(saved):
            image = images[batch_index]
            manifest = build_manifest(
                positive=positive, negative=negative, seed=seed, tag=tag, params=parsed_params,
                outputs=[{
                    "file": result["filename"],
                    "subfolder": result["subfolder"].replace("\\", "/"),
                    "batch_index": batch_index,
                    "batch_size": len(saved),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }],
                prompt_graph=cls.hidden.prompt,
                workflow=workflow,
                include_graph=include_graph,
                include_workflow=include_workflow,
                tool=TOOL,
            )
            image_path = os.path.join(output_dir, result["subfolder"], result["filename"])
            write_json_atomic(sidecar_path(image_path), manifest)
            if first is None:
                first = manifest

        return io.NodeOutput(json.dumps(first, indent=2, default=str), ui=ui.SavedImages(saved))
