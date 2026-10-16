# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""JungleFruit_ApplyCubeLUT: grade images with a 3D .cube LUT (trilinear, torch)."""

from __future__ import annotations

import os
from collections import OrderedDict

import folder_paths
from comfy_api.latest import io

from ..core.lut import CubeLUT, CubeParseError, apply_lut, load_cube
from ..core.paths import UnsafePathError, resolve_inside

LUT_FOLDER = "luts"
BUNDLED_LUTS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "luts"))
NO_LUTS = "(no .cube files found)"
_CACHE: "OrderedDict[tuple[str, float], CubeLUT]" = OrderedDict()
_CACHE_SIZE = 8


def register_lut_folders() -> None:
    """Expose `models/luts` (user LUTs) and the pack's own `luts/` folder to ComfyUI."""
    folder_paths.add_model_folder_path(LUT_FOLDER, os.path.join(folder_paths.models_dir, LUT_FOLDER))
    folder_paths.add_model_folder_path(LUT_FOLDER, BUNDLED_LUTS)


def list_luts() -> list[str]:
    try:
        names = [n for n in folder_paths.get_filename_list(LUT_FOLDER) if n.lower().endswith(".cube")]
    except KeyError:
        names = []
    return names or [NO_LUTS]


def _resolve(lut_name: str, lut_path: str) -> str:
    if lut_path and lut_path.strip():
        roots = folder_paths.get_folder_paths(LUT_FOLDER) + [folder_paths.get_input_directory()]
        try:
            return resolve_inside(lut_path, roots)
        except (UnsafePathError, FileNotFoundError) as exc:
            raise ValueError(f"[Apply .cube LUT] lut_path: {exc}") from None
    if lut_name == NO_LUTS:
        raise ValueError("[Apply .cube LUT] no LUT selected: put .cube files in models/luts")
    path = folder_paths.get_full_path(LUT_FOLDER, lut_name)
    if not path:
        raise ValueError(f"[Apply .cube LUT] LUT not found: {lut_name}")
    return path


def _load_cached(path: str) -> CubeLUT:
    key = (path, os.path.getmtime(path))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    try:
        lut = load_cube(path)
    except CubeParseError as exc:
        raise ValueError(f"[Apply .cube LUT] {os.path.basename(path)}: {exc}") from None
    _CACHE[key] = lut
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return lut


class ApplyCubeLUT(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JungleFruit_ApplyCubeLUT",
            display_name="Apply .cube LUT",
            category="jungle_fruit/color",
            description="Applies a 3D .cube LUT with trilinear interpolation. LUTs are listed from models/luts.",
            search_aliases=["lut", "color grade", "look", "cube"],
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("lut_name", options=list_luts()),
                io.Float.Input("intensity", default=1.0, min=0.0, max=1.0, step=0.01,
                               display_mode=io.NumberDisplay.slider,
                               tooltip="0 = original, 1 = full LUT."),
                io.String.Input("lut_path", default="", advanced=True,
                                tooltip="Optional override: a .cube file inside models/luts or the input folder."),
            ],
            outputs=[io.Image.Output("image")],
        )

    @classmethod
    def fingerprint_inputs(cls, image=None, lut_name="", intensity=1.0, lut_path=""):
        # Re-run if the LUT file is edited on disk.
        try:
            path = _resolve(lut_name, lut_path)
            return (path, os.path.getmtime(path), intensity)
        except Exception as exc:
            return str(exc)

    @classmethod
    def execute(cls, image, lut_name, intensity, lut_path="") -> io.NodeOutput:
        lut = _load_cached(_resolve(lut_name, lut_path))
        return io.NodeOutput(apply_lut(image, lut, intensity))
