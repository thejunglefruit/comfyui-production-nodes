# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Cheap, model-free image quality metrics and a pass/fail gate.

All metrics run on Rec.709 luminance of a ComfyUI IMAGE tensor (B, H, W, C, float 0-1):

- sharpness: variance of the 4-neighbour Laplacian, on a 0-255 scale (same scale as the
  common OpenCV `cv2.Laplacian(gray, CV_64F).var()` heuristic). Blurry or flat frames score low.
- mean_luma: average brightness, 0-1.
- clipped_highlights / crushed_shadows: fraction of pixels at or above / at or below a level.
- contrast: RMS contrast (standard deviation of luminance), 0-0.5.

Nothing here is a perceptual model; the point is to catch obvious failures (black frames,
blown-out frames, mushy renders) in a batch before they reach a human or a client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

REC709 = (0.2126, 0.7152, 0.0722)
NONE_PASS_MODES = ("return_best", "block")


@dataclass(frozen=True)
class Thresholds:
    min_sharpness: float = 50.0
    min_mean_luma: float = 0.08
    max_mean_luma: float = 0.92
    max_clipped_highlights: float = 0.05
    max_crushed_shadows: float = 0.05
    min_contrast: float = 0.04
    highlight_level: float = 0.98
    shadow_level: float = 0.02


@dataclass
class ImageReport:
    index: int
    passed: bool
    sharpness: float
    mean_luma: float
    clipped_highlights: float
    crushed_shadows: float
    contrast: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class GateResult:
    passed: list[int]
    rejected: list[int]
    reports: list[ImageReport]
    fallback_used: bool = False

    def to_dict(self, thresholds: Thresholds) -> dict:
        return {
            "total": len(self.reports),
            "passed": len(self.passed) if not self.fallback_used else 0,
            "rejected": len(self.rejected),
            "fallback_used": self.fallback_used,
            "passed_indices": self.passed,
            "rejected_indices": self.rejected,
            "thresholds": asdict(thresholds),
            "images": [asdict(r) for r in self.reports],
        }


def luminance(images: torch.Tensor) -> torch.Tensor:
    """(B, H, W, C) -> (B, H, W). Grayscale input (C == 1) is returned as is; alpha is ignored."""
    if images.ndim != 4:
        raise ValueError(f"expected an IMAGE tensor (B, H, W, C), got shape {tuple(images.shape)}")
    images = images.float()
    if images.shape[-1] == 1:
        return images[..., 0]
    if images.shape[-1] < 3:
        raise ValueError(f"expected 1, 3 or 4 channels, got {images.shape[-1]}")
    weights = torch.tensor(REC709, dtype=images.dtype, device=images.device)
    return (images[..., :3] * weights).sum(dim=-1)


_LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def laplacian_variance(luma: torch.Tensor) -> torch.Tensor:
    """Per-image variance of the Laplacian on the 0-255 scale. (B, H, W) -> (B,)"""
    b, h, w = luma.shape
    if h < 3 or w < 3:
        return torch.zeros(b, device=luma.device)
    kernel = _LAPLACIAN.to(device=luma.device, dtype=luma.dtype).view(1, 1, 3, 3)
    response = F.conv2d(luma.unsqueeze(1) * 255.0, kernel)  # 'valid' conv: no border artefacts
    return response.flatten(1).var(dim=1, unbiased=False)


def compute_metrics(images: torch.Tensor, thresholds: Thresholds = Thresholds()) -> list[dict]:
    """Metrics for every image in the batch, as plain floats."""
    luma = luminance(images).clamp(0.0, 1.0)
    flat = luma.flatten(1)
    sharp = laplacian_variance(luma)
    stats = {
        "sharpness": sharp,
        "mean_luma": flat.mean(dim=1),
        "clipped_highlights": (flat >= thresholds.highlight_level).float().mean(dim=1),
        "crushed_shadows": (flat <= thresholds.shadow_level).float().mean(dim=1),
        "contrast": flat.std(dim=1, unbiased=False),
    }
    return [{k: round(float(v[i]), 6) for k, v in stats.items()} for i in range(luma.shape[0])]


def judge(metrics: dict, t: Thresholds) -> list[str]:
    """Return human-readable reasons for failure; an empty list means the image passes."""
    reasons = []
    if metrics["sharpness"] < t.min_sharpness:
        reasons.append(f"soft: sharpness {metrics['sharpness']:.1f} < {t.min_sharpness:g}")
    if metrics["mean_luma"] < t.min_mean_luma:
        reasons.append(f"underexposed: mean luma {metrics['mean_luma']:.3f} < {t.min_mean_luma:g}")
    if metrics["mean_luma"] > t.max_mean_luma:
        reasons.append(f"overexposed: mean luma {metrics['mean_luma']:.3f} > {t.max_mean_luma:g}")
    if metrics["clipped_highlights"] > t.max_clipped_highlights:
        reasons.append(f"clipped highlights: {metrics['clipped_highlights']:.1%} > {t.max_clipped_highlights:.1%}")
    if metrics["crushed_shadows"] > t.max_crushed_shadows:
        reasons.append(f"crushed shadows: {metrics['crushed_shadows']:.1%} > {t.max_crushed_shadows:.1%}")
    if metrics["contrast"] < t.min_contrast:
        reasons.append(f"flat: contrast {metrics['contrast']:.3f} < {t.min_contrast:g}")
    return reasons


def _rank_key(report: ImageReport) -> tuple:
    """Lower is better: fewest failed checks first, then sharpest, then closest to mid exposure."""
    return (len(report.reasons), -report.sharpness, abs(report.mean_luma - 0.5))


def run_gate(images: torch.Tensor, thresholds: Thresholds = Thresholds(), if_none_pass: str = "return_best") -> GateResult:
    """Split a batch into passed / rejected indices.

    If nothing passes and `if_none_pass == "return_best"`, the single best image is put in
    `passed` and `fallback_used` is set, so a downstream save or upscale never receives an
    empty batch. With "block" the passed list stays empty and the caller decides what to do.
    """
    if if_none_pass not in NONE_PASS_MODES:
        raise ValueError(f"if_none_pass must be one of {NONE_PASS_MODES}")
    reports = []
    for i, m in enumerate(compute_metrics(images, thresholds)):
        reasons = judge(m, thresholds)
        reports.append(ImageReport(index=i, passed=not reasons, reasons=reasons, **m))
    passed = [r.index for r in reports if r.passed]
    rejected = [r.index for r in reports if not r.passed]
    fallback = False
    if not passed and reports and if_none_pass == "return_best":
        best = min(reports, key=_rank_key)
        passed, fallback = [best.index], True
    return GateResult(passed=passed, rejected=rejected, reports=reports, fallback_used=fallback)
