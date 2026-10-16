# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
import pytest
import torch
import torch.nn.functional as F

from jf_nodes.core.quality import Thresholds, compute_metrics, judge, laplacian_variance, luminance, run_gate


def checkerboard(size=128, cell=4):
    """Mid-grey checkerboard (0.25 / 0.75): sharp, well exposed, contrasty."""
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    board = (((x // cell) + (y // cell)) % 2).float() * 0.5 + 0.25
    return board.unsqueeze(-1).expand(size, size, 3).unsqueeze(0).contiguous()


def blur(images, passes=6):
    x = images.permute(0, 3, 1, 2)
    kernel = torch.ones(3, 1, 5, 5) / 25.0
    for _ in range(passes):
        x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), kernel, groups=3)
    return x.permute(0, 2, 3, 1)


def test_luminance_weights_rec709():
    img = torch.zeros(1, 1, 3, 3)
    img[0, 0, 0, 0] = img[0, 0, 1, 1] = img[0, 0, 2, 2] = 1.0
    assert torch.allclose(luminance(img)[0, 0], torch.tensor([0.2126, 0.7152, 0.0722]))


def test_flat_image_has_zero_sharpness():
    assert laplacian_variance(torch.full((2, 32, 32), 0.5)).tolist() == [0.0, 0.0]


def test_blurry_scores_far_below_sharp():
    sharp = checkerboard()
    soft = blur(sharp)
    s_sharp, s_soft = (compute_metrics(x)[0]["sharpness"] for x in (sharp, soft))
    assert s_sharp > 1000 and s_soft < s_sharp / 20


def test_blown_out_frame_is_flagged():
    burnt = torch.full((1, 64, 64, 3), 1.0)
    m = compute_metrics(burnt)[0]
    assert m["clipped_highlights"] == 1.0 and m["mean_luma"] == 1.0
    reasons = " ".join(judge(m, Thresholds()))
    assert "overexposed" in reasons and "clipped highlights" in reasons


def test_black_frame_is_flagged():
    m = compute_metrics(torch.zeros(1, 64, 64, 3))[0]
    reasons = " ".join(judge(m, Thresholds()))
    assert "underexposed" in reasons and "crushed shadows" in reasons and "flat" in reasons


def test_gate_splits_batch_and_reports_reasons():
    sharp = checkerboard()
    batch = torch.cat([sharp, blur(sharp), torch.ones_like(sharp)], dim=0)
    result = run_gate(batch, Thresholds())
    assert result.passed == [0]
    assert result.rejected == [1, 2]
    assert not result.fallback_used
    assert any("soft" in r for r in result.reports[1].reasons)
    report = result.to_dict(Thresholds())
    assert report["total"] == 3 and report["passed"] == 1 and len(report["images"]) == 3


def test_none_pass_return_best_picks_least_bad():
    sharp = checkerboard()
    strict = Thresholds(min_sharpness=1e9)  # nothing can pass
    batch = torch.cat([blur(sharp), sharp], dim=0)
    result = run_gate(batch, strict, "return_best")
    assert result.fallback_used and result.passed == [1] and result.rejected == [0, 1]
    assert result.to_dict(strict)["passed"] == 0


def test_none_pass_block_returns_empty():
    result = run_gate(checkerboard(), Thresholds(min_sharpness=1e9), "block")
    assert result.passed == [] and not result.fallback_used


def test_rgba_and_grayscale_inputs():
    rgba = torch.cat([checkerboard(), torch.ones(1, 128, 128, 1)], dim=-1)
    assert compute_metrics(rgba)[0]["sharpness"] > 1000
    gray = checkerboard()[..., :1]
    assert compute_metrics(gray)[0]["sharpness"] > 1000


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        luminance(torch.zeros(64, 64, 3))
