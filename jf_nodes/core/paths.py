# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Path helpers that keep user-supplied paths inside allowed folders.

A ComfyUI server is often reachable by more than one person (a studio LAN, a cloud pod),
so any path typed into a widget is treated as untrusted: it must resolve inside one of
the folders the server already exposes.
"""

from __future__ import annotations

import os
from typing import Iterable


class UnsafePathError(ValueError):
    """Raised when a user path escapes every allowed root."""


def _is_within(root: str, candidate: str) -> bool:
    root = os.path.normcase(os.path.realpath(root))
    candidate = os.path.normcase(os.path.realpath(candidate))
    try:
        return os.path.commonpath([root, candidate]) == root
    except ValueError:  # different drives on Windows
        return False


def resolve_inside(user_path: str, roots: Iterable[str]) -> str:
    """Resolve `user_path` against each root and return the first existing file inside a root.

    Relative paths are joined to each root in order. Absolute paths are accepted only when
    they already point inside one of the roots.
    """
    user_path = (user_path or "").strip().strip('"').strip("'")
    if not user_path:
        raise UnsafePathError("empty path")
    roots = [r for r in roots if r]
    candidates = [user_path] if os.path.isabs(user_path) else [os.path.join(r, user_path) for r in roots]
    inside_any = False
    for cand in candidates:
        for root in roots:
            if _is_within(root, cand):
                inside_any = True
                if os.path.isfile(cand):
                    return os.path.realpath(cand)
    if not inside_any:
        raise UnsafePathError(f"'{user_path}' is outside the allowed folders")
    raise FileNotFoundError(f"file not found: '{user_path}' (looked in {len(roots)} allowed folder(s))")
