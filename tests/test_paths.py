# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
import os

import pytest

from jf_nodes.core.paths import UnsafePathError, resolve_inside


def test_relative_file_inside_root(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "prompts.csv"
    f.write_text("x")
    assert resolve_inside("sub/prompts.csv", [str(tmp_path)]) == os.path.realpath(f)


def test_traversal_is_rejected(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("x")
    with pytest.raises(UnsafePathError):
        resolve_inside("../secret.txt", [str(root)])


def test_absolute_outside_is_rejected(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    outside = tmp_path / "other.csv"
    outside.write_text("x")
    with pytest.raises(UnsafePathError):
        resolve_inside(str(outside), [str(root)])


def test_absolute_inside_second_root_is_accepted(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    f = b / "look.cube"
    f.write_text("x")
    assert resolve_inside(str(f), [str(a), str(b)]) == os.path.realpath(f)


def test_missing_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_inside("nope.csv", [str(tmp_path)])
    with pytest.raises(UnsafePathError):
        resolve_inside("  ", [str(tmp_path)])
