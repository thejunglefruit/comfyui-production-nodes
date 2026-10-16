# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
from pathlib import Path

import numpy as np
import pytest
import torch

from jf_nodes.core.lut import CubeParseError, apply_lut, identity_lut, load_cube, lut_from_function, parse_cube, to_cube_text

ROOT = Path(__file__).resolve().parents[1]


def random_image(b=2, h=17, w=23, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, h, w, c, generator=g)


def reference_trilinear(table: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Plain numpy trilinear lookup, table indexed [b, g, r]. Independent of grid_sample."""
    n = table.shape[0]
    p = np.clip(rgb, 0, 1) * (n - 1)
    i0 = np.clip(np.floor(p).astype(int), 0, n - 2)
    f = p - i0
    out = np.zeros_like(rgb)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = ((f[:, 0] if dr else 1 - f[:, 0]) * (f[:, 1] if dg else 1 - f[:, 1]) * (f[:, 2] if db else 1 - f[:, 2]))
                out += w[:, None] * table[i0[:, 2] + db, i0[:, 1] + dg, i0[:, 0] + dr]
    return out


def test_identity_lut_leaves_image_unchanged():
    img = random_image()
    out = apply_lut(img, identity_lut(33))
    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-5)


def test_identity_survives_cube_round_trip():
    lut = parse_cube(to_cube_text(identity_lut(17)))
    assert lut.size == 17 and lut.title == "identity"
    img = random_image()
    assert torch.allclose(apply_lut(img, lut), img, atol=1e-5)


def test_known_linear_lut_invert():
    invert = lut_from_function(lambda rgb: 1.0 - rgb, size=9, title="invert")
    img = random_image()
    assert torch.allclose(apply_lut(img, invert), 1.0 - img, atol=1e-5)


def test_channel_order_red_fastest():
    # A LUT that outputs (r, r, r): if r/b were swapped the result would follow blue instead.
    lut = lut_from_function(lambda rgb: rgb[..., :1].expand_as(rgb), size=5)
    img = torch.tensor([[[[0.8, 0.1, 0.3]]]])
    assert torch.allclose(apply_lut(img, lut), torch.full((1, 1, 1, 3), 0.8), atol=1e-5)


def test_matches_independent_trilinear_reference_on_nonlinear_lut():
    lut = lut_from_function(lambda rgb: rgb ** 2.2, size=7)
    img = random_image(b=1, h=31, w=29)
    expected = reference_trilinear(lut.table.numpy(), img[0].reshape(-1, 3).numpy()).reshape(img[0].shape)
    assert np.allclose(apply_lut(img, lut)[0].numpy(), expected, atol=1e-5)


def test_intensity_blends_and_alpha_passes_through():
    invert = lut_from_function(lambda rgb: 1.0 - rgb, size=5)
    img = random_image(c=4)
    half = apply_lut(img, invert, intensity=0.5)
    assert torch.allclose(half[..., :3], torch.full_like(img[..., :3], 0.5), atol=1e-5)
    assert torch.equal(half[..., 3], img[..., 3])
    assert torch.equal(apply_lut(img, invert, intensity=0.0), img)


def test_domain_min_max_is_respected():
    text = to_cube_text(identity_lut(5)).replace("DOMAIN_MAX 1 1 1", "DOMAIN_MAX 2 2 2")
    lut = parse_cube(text)
    img = torch.full((1, 2, 2, 3), 1.0)
    assert torch.allclose(apply_lut(img, lut), torch.full_like(img, 0.5), atol=1e-5)


def test_chunking_gives_same_result():
    lut = lut_from_function(lambda rgb: rgb.flip(-1), size=9)
    img = random_image()
    assert torch.allclose(apply_lut(img, lut), apply_lut(img, lut, chunk_pixels=7), atol=1e-6)


def test_parses_comments_title_and_input_range():
    body = "\n".join(f"{r} {g} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1))
    lut = parse_cube(f'# comment\nTITLE "My Look"\nLUT_3D_SIZE 2\nLUT_3D_INPUT_RANGE 0 1\n{body}\n')
    assert lut.title == "My Look" and lut.size == 2 and lut.domain_max == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("text, needle", [
    ("0 0 0\n", "missing LUT_3D_SIZE"),
    ("LUT_1D_SIZE 4\n", "1D LUTs are not supported"),
    ("LUT_3D_SIZE 2\n0 0 0\n", "expected 8 RGB rows"),
    ("LUT_3D_SIZE 1\n", "outside"),
    ("LUT_3D_SIZE 2\n0 0\n", "needs 3 numbers"),
    ("LUT_3D_SIZE 2\nDOMAIN_MIN 1 1 1\nDOMAIN_MAX 0 0 0\n" + "0 0 0\n" * 8, "DOMAIN_MAX must be greater"),
])
def test_malformed_files_raise_clear_errors(text, needle):
    with pytest.raises(CubeParseError, match=needle):
        parse_cube(text)


@pytest.mark.parametrize("name", ["teal_orange", "bleach_bypass"])
def test_bundled_luts_load_and_change_the_image(name):
    lut = load_cube(str(ROOT / "luts" / f"{name}.cube"))
    assert lut.size == 33
    img = random_image()
    out = apply_lut(img, lut)
    assert out.min() >= 0 and out.max() <= 1
    assert (out - img).abs().mean() > 0.01
