# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Read, write and apply 3D LUTs in the Adobe/Resolve `.cube` format.

Supported keywords: TITLE, LUT_3D_SIZE, DOMAIN_MIN, DOMAIN_MAX, LUT_3D_INPUT_RANGE.
Data order follows the spec: red changes fastest, then green, then blue. The table is kept
as a (N, N, N, 3) tensor indexed [blue, green, red], which is exactly the (D, H, W) layout
`torch.nn.functional.grid_sample` expects, so trilinear lookup is a single call.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

import torch
import torch.nn.functional as F

MIN_SIZE, MAX_SIZE = 2, 256


class CubeParseError(ValueError):
    """The file is not a usable 3D .cube LUT."""


@dataclass
class CubeLUT:
    title: str
    size: int
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]
    table: torch.Tensor  # (size, size, size, 3) float32, indexed [b, g, r]


def _floats(parts: list[str], keyword: str, count: int, line_no: int) -> list[float]:
    if len(parts) != count:
        raise CubeParseError(f"line {line_no}: {keyword} needs {count} numbers")
    try:
        return [float(p) for p in parts]
    except ValueError:
        raise CubeParseError(f"line {line_no}: {keyword} has a non-numeric value") from None


def parse_cube(text: str) -> CubeLUT:
    """Parse .cube text. Raises CubeParseError with a line number on malformed input."""
    title, size = "", None
    dmin, dmax = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    values: list[float] = []
    for line_no, raw in enumerate(text.lstrip("﻿").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split(None, 1)[0].upper()
        if head[0].isdigit() or head[0] in "+-.":
            values.extend(_floats(line.split(), "RGB row", 3, line_no))
        elif head == "TITLE":
            rest = line[len("TITLE"):].strip()
            try:
                title = " ".join(shlex.split(rest)) if rest else ""
            except ValueError:
                title = rest.strip('"')
        elif head == "LUT_3D_SIZE":
            (n,) = _floats(line.split()[1:], head, 1, line_no)
            size = int(n)
            if not MIN_SIZE <= size <= MAX_SIZE:
                raise CubeParseError(f"line {line_no}: LUT_3D_SIZE {size} is outside {MIN_SIZE}-{MAX_SIZE}")
        elif head == "LUT_1D_SIZE":
            raise CubeParseError("1D LUTs are not supported by this node; export a 3D .cube instead")
        elif head == "DOMAIN_MIN":
            dmin = _floats(line.split()[1:], head, 3, line_no)
        elif head == "DOMAIN_MAX":
            dmax = _floats(line.split()[1:], head, 3, line_no)
        elif head == "LUT_3D_INPUT_RANGE":
            lo, hi = _floats(line.split()[1:], head, 2, line_no)
            dmin, dmax = [lo] * 3, [hi] * 3
        else:
            continue  # unknown keywords are ignored, as most hosts do
    if size is None:
        raise CubeParseError("missing LUT_3D_SIZE (is this a 3D .cube file?)")
    expected = size ** 3
    if len(values) != expected * 3:
        raise CubeParseError(f"expected {expected} RGB rows for LUT_3D_SIZE {size}, found {len(values) // 3}")
    if any(hi <= lo for lo, hi in zip(dmin, dmax)):
        raise CubeParseError("DOMAIN_MAX must be greater than DOMAIN_MIN on every channel")
    table = torch.tensor(values, dtype=torch.float32).view(size, size, size, 3)
    return CubeLUT(title=title, size=size, domain_min=tuple(dmin), domain_max=tuple(dmax), table=table)


def load_cube(path: str) -> CubeLUT:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_cube(fh.read())


def identity_lut(size: int = 33) -> CubeLUT:
    """A LUT that maps every colour to itself (useful for tests and as a starting grid)."""
    ramp = torch.linspace(0.0, 1.0, size)
    b, g, r = torch.meshgrid(ramp, ramp, ramp, indexing="ij")
    table = torch.stack([r, g, b], dim=-1)
    return CubeLUT(title="identity", size=size, domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0), table=table)


def lut_from_function(fn, size: int = 33, title: str = "") -> CubeLUT:
    """Bake a colour transform into a LUT. `fn` maps an (..., 3) RGB tensor in 0-1 to RGB."""
    lut = identity_lut(size)
    lut.table = fn(lut.table).clamp(0.0, 1.0).float()
    lut.title = title
    return lut


def to_cube_text(lut: CubeLUT, decimals: int = 6) -> str:
    """Serialise a LUT back to .cube text (red fastest)."""
    lines = []
    if lut.title:
        lines.append(f'TITLE "{lut.title}"')
    lines.append(f"LUT_3D_SIZE {lut.size}")
    lines.append("DOMAIN_MIN " + " ".join(f"{v:g}" for v in lut.domain_min))
    lines.append("DOMAIN_MAX " + " ".join(f"{v:g}" for v in lut.domain_max))
    fmt = f"{{:.{decimals}f}} {{:.{decimals}f}} {{:.{decimals}f}}"
    for r, g, b in lut.table.reshape(-1, 3).tolist():
        lines.append(fmt.format(r, g, b))
    return "\n".join(lines) + "\n"


def apply_lut(images: torch.Tensor, lut: CubeLUT, intensity: float = 1.0, chunk_pixels: int = 4_194_304) -> torch.Tensor:
    """Apply `lut` to an IMAGE tensor (B, H, W, C) with trilinear interpolation.

    `intensity` blends between the original (0) and the fully graded image (1).
    Alpha (4th channel) is passed through untouched. Work is chunked per image so a large
    batch does not allocate one giant sampling grid.
    """
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError(f"expected an RGB/RGBA IMAGE tensor (B, H, W, 3|4), got {tuple(images.shape)}")
    intensity = float(min(max(intensity, 0.0), 1.0))
    if intensity == 0.0:
        return images.clone()

    device, out_dtype = images.device, images.dtype
    table = lut.table.to(device=device, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,D=b,H=g,W=r)
    dmin = torch.tensor(lut.domain_min, device=device, dtype=torch.float32)
    span = torch.tensor(lut.domain_max, device=device, dtype=torch.float32) - dmin

    out = images.clone()
    for i in range(images.shape[0]):
        rgb = images[i, ..., :3].float()
        h, w, _ = rgb.shape
        flat = rgb.reshape(-1, 3)
        graded = torch.empty_like(flat)
        for start in range(0, flat.shape[0], chunk_pixels):
            px = flat[start:start + chunk_pixels]
            coords = ((px - dmin) / span).clamp(0.0, 1.0) * 2.0 - 1.0  # grid order (x=r, y=g, z=b)
            grid = coords.view(1, 1, 1, -1, 3)
            sampled = F.grid_sample(table, grid, mode="bilinear", padding_mode="border", align_corners=True)
            graded[start:start + chunk_pixels] = sampled.view(3, -1).t()
        graded = graded.view(h, w, 3)
        if intensity < 1.0:
            graded = rgb + (graded - rgb) * intensity
        out[i, ..., :3] = graded.clamp(0.0, 1.0).to(out_dtype)
    return out
