# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""Parse a prompt sheet (CSV) and pick one row per queued run.

Required columns: prompt, negative, seed.
Optional columns: cfg, steps, width, height, tag (empty cells fall back to defaults).
Headers are case-insensitive; comma, semicolon and tab delimiters are detected.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

REQUIRED_COLUMNS = ("prompt", "negative", "seed")
OPTIONAL_COLUMNS = ("cfg", "steps", "width", "height", "tag")
MAX_SEED = 0xFFFFFFFFFFFFFFFF
OVERFLOW_MODES = ("wrap", "clamp", "error")


class CsvBatchError(ValueError):
    """A problem with the prompt sheet, phrased so an artist can fix it."""


@dataclass(frozen=True)
class RowDefaults:
    cfg: float = 7.0
    steps: int = 20
    width: int = 1024
    height: int = 1024
    tag: str = ""


@dataclass(frozen=True)
class PromptRow:
    prompt: str
    negative: str
    seed: int
    cfg: float
    steps: int
    width: int
    height: int
    tag: str
    line: int  # 1-based line in the source text, for error messages and reports


def _detect_dialect(sample: str) -> type[csv.Dialect] | csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def _to_int(value: str, column: str, line: int, lo: int, hi: int) -> int:
    try:
        out = int(value)  # exact, also for 64-bit seeds
    except ValueError:
        try:
            number = float(value)  # spreadsheets often export "20.0"
            if not number.is_integer():
                raise ValueError
            out = int(number)
        except ValueError:
            out = None
    if out is None:
        raise CsvBatchError(f"line {line}: column '{column}' must be an integer, got '{value}'") from None
    if not lo <= out <= hi:
        raise CsvBatchError(f"line {line}: column '{column}' must be between {lo} and {hi}, got {out}")
    return out


def _to_float(value: str, column: str, line: int) -> float:
    try:
        return float(value)
    except ValueError:
        raise CsvBatchError(f"line {line}: column '{column}' must be a number, got '{value}'") from None


def parse_prompt_csv(text: str, defaults: RowDefaults | None = None) -> list[PromptRow]:
    """Parse CSV text into validated rows. Raises CsvBatchError with a line number on bad data."""
    defaults = defaults or RowDefaults()
    text = (text or "").lstrip("﻿")
    if not text.strip():
        raise CsvBatchError("the CSV is empty")

    reader = csv.reader(io.StringIO(text), _detect_dialect(text[:4096]))
    try:
        header = next(reader)
    except StopIteration:
        raise CsvBatchError("the CSV is empty") from None
    columns = [h.strip().lower() for h in header]
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        raise CsvBatchError(
            f"missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(c for c in columns if c) or '(none)'}. "
            f"Required: {', '.join(REQUIRED_COLUMNS)}; optional: {', '.join(OPTIONAL_COLUMNS)}"
        )
    position = {name: columns.index(name) for name in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if name in columns}

    rows: list[PromptRow] = []
    for cells in reader:
        line = reader.line_num
        if not any(c.strip() for c in cells):
            continue  # blank line

        def cell(name: str) -> str:
            i = position.get(name)
            return cells[i].strip() if i is not None and i < len(cells) else ""

        prompt = cell("prompt")
        if not prompt:
            raise CsvBatchError(f"line {line}: 'prompt' is empty")
        seed_text = cell("seed")
        if not seed_text:
            raise CsvBatchError(f"line {line}: 'seed' is empty")

        rows.append(PromptRow(
            prompt=prompt,
            negative=cell("negative"),
            seed=_to_int(seed_text, "seed", line, 0, MAX_SEED),
            cfg=_to_float(cell("cfg"), "cfg", line) if cell("cfg") else defaults.cfg,
            steps=_to_int(cell("steps"), "steps", line, 1, 10000) if cell("steps") else defaults.steps,
            width=_to_int(cell("width"), "width", line, 16, 16384) if cell("width") else defaults.width,
            height=_to_int(cell("height"), "height", line, 16, 16384) if cell("height") else defaults.height,
            tag=cell("tag") or defaults.tag,
            line=line,
        ))
    if not rows:
        raise CsvBatchError("the CSV has a header but no data rows")
    return rows


def select_row(rows: list[PromptRow], index: int, on_overflow: str = "wrap") -> tuple[PromptRow, int]:
    """Return (row, effective_index). `on_overflow` decides what happens past the last row."""
    if not rows:
        raise CsvBatchError("no rows to select from")
    if on_overflow not in OVERFLOW_MODES:
        raise ValueError(f"on_overflow must be one of {OVERFLOW_MODES}")
    if index < 0:
        raise CsvBatchError(f"row index must be >= 0, got {index}")
    n = len(rows)
    if index >= n:
        if on_overflow == "error":
            raise CsvBatchError(f"row index {index} is past the end of the sheet ({n} rows, last index {n - 1})")
        index = index % n if on_overflow == "wrap" else n - 1
    return rows[index], index


__all__ = [
    "CsvBatchError", "PromptRow", "RowDefaults", "parse_prompt_csv", "select_row",
    "REQUIRED_COLUMNS", "OPTIONAL_COLUMNS", "OVERFLOW_MODES",
]
