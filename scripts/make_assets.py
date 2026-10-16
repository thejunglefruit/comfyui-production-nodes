# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Generate the bundled example LUTs and the README before/after image.

Everything is computed from formulas in this file (no third-party LUTs, no photos), so the
outputs carry the same licence as the code.

    python scripts/make_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jf_nodes.core.lut import apply_lut, lut_from_function, to_cube_text  # noqa: E402

REC709 = torch.tensor([0.2126, 0.7152, 0.0722])
SIZE = 33


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb * REC709).sum(-1, keepdim=True)


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def teal_orange(rgb: torch.Tensor) -> torch.Tensor:
    """Split-tone: cool teal shadows, warm orange highlights, gentle S-curve, a touch more saturation."""
    y = _luma(rgb)
    teal = torch.tensor([0.0, 0.50, 0.55])
    orange = torch.tensor([1.0, 0.55, 0.20])
    shadows = 1.0 - _smoothstep(0.05, 0.55, y)
    highlights = _smoothstep(0.35, 0.95, y)
    out = rgb + 0.18 * shadows * (teal - y) + 0.14 * highlights * (orange - y)
    out = y + (out - y) * 1.12                                   # saturation
    out = out.clamp(0, 1)
    return out + 0.08 * torch.sin((out - 0.5) * torch.pi) * (1 - (2 * out - 1).abs())  # soft S-curve


def bleach_bypass(rgb: torch.Tensor) -> torch.Tensor:
    """Film bleach-bypass look: overlay the luminance on itself, then pull saturation down."""
    y = _luma(rgb).expand_as(rgb)
    overlay = torch.where(rgb < 0.5, 2 * rgb * y, 1 - 2 * (1 - rgb) * (1 - y))
    mixed = rgb + (overlay - rgb) * 0.7
    my = _luma(mixed)
    return my + (mixed - my) * 0.45


def test_chart(h: int = 256, w: int = 384) -> torch.Tensor:
    """Synthetic chart: hue sweep across, brightness down the frame, a grey ramp and warm patches."""
    x = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    yy = torch.linspace(1, 0, h).view(h, 1).expand(h, w)
    hue = x * 6.0
    k = torch.stack([(5 + hue) % 6, (3 + hue) % 6, (1 + hue) % 6], -1)
    rgb = 1 - (torch.minimum(k, 4 - k).clamp(0, 1))       # HSV hue wheel at full saturation
    sat = 0.75
    rgb = (1 - sat) + sat * rgb
    img = rgb * (0.08 + 0.9 * yy.unsqueeze(-1))
    band = h // 8
    img[-band:] = x[:band].unsqueeze(-1).expand(band, w, 3)  # neutral grey ramp
    for i, tone in enumerate([(0.93, 0.76, 0.64), (0.78, 0.57, 0.44), (0.55, 0.38, 0.28)]):
        x0 = 24 + i * 56
        img[24:72, x0:x0 + 44] = torch.tensor(tone)        # skin-like tones
    return img.unsqueeze(0)


def save_png(tensor: torch.Tensor, path: Path) -> None:
    from PIL import Image
    arr = (tensor.clamp(0, 1) * 255).round().byte().numpy()
    Image.fromarray(arr).save(path)  # plain PNG, no metadata chunks


def main() -> None:
    lut_dir = ROOT / "luts"
    img_dir = ROOT / "docs" / "images"
    lut_dir.mkdir(exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    looks = {
        "teal_orange": lut_from_function(teal_orange, SIZE, "JF Teal Orange"),
        "bleach_bypass": lut_from_function(bleach_bypass, SIZE, "JF Bleach Bypass"),
    }
    for name, lut in looks.items():  # 4 decimals: well below one 10-bit code value, smaller files
        (lut_dir / f"{name}.cube").write_text(to_cube_text(lut, decimals=4), encoding="utf-8", newline="\n")
        print(f"wrote luts/{name}.cube ({lut.size}^3)")

    chart = test_chart()
    gap = torch.ones(1, chart.shape[1], 8, 3)
    panels = [chart, gap, apply_lut(chart, looks["teal_orange"]), gap, apply_lut(chart, looks["bleach_bypass"])]
    save_png(torch.cat(panels, dim=2)[0], img_dir / "lut_before_after.png")
    print("wrote docs/images/lut_before_after.png  (original | teal_orange | bleach_bypass)")


if __name__ == "__main__":
    main()
