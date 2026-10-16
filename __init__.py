# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""comfyui-production-nodes: batch, quality control, look and provenance nodes for ComfyUI (V3 API)."""

try:
    from comfy_api.latest import ComfyExtension, io
except ImportError:
    # Imported outside ComfyUI (pytest collects the repo root as a package). The pure logic in
    # jf_nodes.core is tested directly, so there is nothing to register here.
    ComfyExtension = None

if ComfyExtension is not None:
    from typing_extensions import override

    from .jf_nodes.nodes import NODES, register_lut_folders

    register_lut_folders()

    class ProductionNodesExtension(ComfyExtension):
        @override
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return list(NODES)

    async def comfy_entrypoint() -> ProductionNodesExtension:
        return ProductionNodesExtension()
