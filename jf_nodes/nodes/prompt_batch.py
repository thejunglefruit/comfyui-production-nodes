# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""JungleFruit_PromptBatchCSV: one CSV row per queued run."""

from __future__ import annotations

import hashlib

import folder_paths
from comfy_api.latest import io, ui

from ..core.csv_batch import OVERFLOW_MODES, CsvBatchError, RowDefaults, parse_prompt_csv, select_row
from ..core.paths import UnsafePathError, resolve_inside


def _read_source(csv_text: str, csv_path: str) -> str:
    """Pasted text wins; otherwise read a file that must live inside ComfyUI's input folder."""
    if csv_text and csv_text.strip():
        return csv_text
    if not (csv_path and csv_path.strip()):
        raise CsvBatchError("give either csv_text or csv_path (a file inside ComfyUI's input folder)")
    try:
        path = resolve_inside(csv_path, [folder_paths.get_input_directory()])
    except (UnsafePathError, FileNotFoundError) as exc:
        raise CsvBatchError(f"csv_path: {exc}") from None
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        return fh.read()


class PromptBatchCSV(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JungleFruit_PromptBatchCSV",
            display_name="Prompt Batch from CSV",
            category="jungle_fruit/batch",
            description=(
                "Reads a prompt sheet (columns: prompt, negative, seed; optional cfg, steps, width, height, tag) "
                "and outputs one row. Set the index control to 'increment' and queue N runs to render the whole sheet."
            ),
            search_aliases=["csv", "prompt list", "batch prompts", "spreadsheet"],
            inputs=[
                io.String.Input("csv_path", default="prompts.csv",
                                tooltip="CSV file inside ComfyUI's input folder (ignored when csv_text is filled)."),
                io.String.Input("csv_text", multiline=True, default="",
                                placeholder="prompt,negative,seed\n\"a red fox\",\"blurry\",42",
                                tooltip="Paste CSV here to skip the file."),
                io.Int.Input("index", default=0, min=0, max=1_000_000,
                             control_after_generate=io.ControlAfterGenerate.increment,
                             tooltip="Row to output (0 = first data row)."),
                io.Combo.Input("on_overflow", options=list(OVERFLOW_MODES), default="wrap",
                               tooltip="Past the last row: wrap to the start, clamp to the last row, or stop with an error."),
                io.Float.Input("default_cfg", default=7.0, min=0.0, max=100.0, step=0.1, advanced=True),
                io.Int.Input("default_steps", default=20, min=1, max=10000, advanced=True),
                io.Int.Input("default_width", default=1024, min=16, max=16384, step=8, advanced=True),
                io.Int.Input("default_height", default=1024, min=16, max=16384, step=8, advanced=True),
            ],
            outputs=[
                io.String.Output("prompt"),
                io.String.Output("negative"),
                io.Int.Output("seed"),
                io.Float.Output("cfg"),
                io.Int.Output("steps"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.String.Output("tag"),
                io.Int.Output("row_index"),
                io.Int.Output("total_rows"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, csv_path="", csv_text="", index=0, on_overflow="wrap", **kwargs):
        # Re-run when the file on disk changes, not only when widget values change.
        try:
            content = _read_source(csv_text, csv_path)
        except Exception as exc:  # surface the real error at execute time
            content = f"error:{exc}"
        digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        return (digest, index, on_overflow, tuple(sorted(kwargs.items())))

    @classmethod
    def execute(cls, csv_path, csv_text, index, on_overflow,
                default_cfg=7.0, default_steps=20, default_width=1024, default_height=1024) -> io.NodeOutput:
        defaults = RowDefaults(cfg=default_cfg, steps=default_steps, width=default_width, height=default_height)
        try:
            rows = parse_prompt_csv(_read_source(csv_text, csv_path), defaults)
            row, effective = select_row(rows, index, on_overflow)
        except CsvBatchError as exc:
            raise ValueError(f"[Prompt Batch from CSV] {exc}") from None
        summary = f"row {effective + 1}/{len(rows)} (line {row.line})" + (f" [{row.tag}]" if row.tag else "")
        return io.NodeOutput(
            row.prompt, row.negative, row.seed, float(row.cfg), row.steps, row.width, row.height,
            row.tag, effective, len(rows),
            ui=ui.PreviewText(summary),
        )
