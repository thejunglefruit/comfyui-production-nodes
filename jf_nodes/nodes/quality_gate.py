# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""JungleFruit_QualityGate: split a batch into keepers and rejects with cheap metrics."""

from __future__ import annotations

import json

from comfy_api.latest import io, ui
from comfy_execution.graph_utils import ExecutionBlocker

from ..core.quality import NONE_PASS_MODES, Thresholds, run_gate


class QualityGate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="JungleFruit_QualityGate",
            display_name="Quality Gate",
            category="jungle_fruit/quality",
            description=(
                "Scores every image (sharpness, exposure, clipping, contrast) without loading any model, "
                "then routes keepers to 'passed' and failures to 'rejected', with a JSON report."
            ),
            search_aliases=["qc", "filter", "blur detection", "exposure check"],
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("min_sharpness", default=50.0, min=0.0, max=100000.0, step=1.0,
                               tooltip="Variance of the Laplacian (0-255 scale). Raise for crisp product shots, lower for soft looks."),
                io.Float.Input("min_brightness", default=0.08, min=0.0, max=1.0, step=0.01,
                               tooltip="Minimum mean luminance (0-1)."),
                io.Float.Input("max_brightness", default=0.92, min=0.0, max=1.0, step=0.01,
                               tooltip="Maximum mean luminance (0-1)."),
                io.Float.Input("max_clipped_highlights", default=0.05, min=0.0, max=1.0, step=0.005,
                               tooltip="Max fraction of pixels at >= 98% luminance."),
                io.Float.Input("max_crushed_shadows", default=0.05, min=0.0, max=1.0, step=0.005,
                               tooltip="Max fraction of pixels at <= 2% luminance."),
                io.Float.Input("min_contrast", default=0.04, min=0.0, max=0.5, step=0.005,
                               tooltip="Minimum RMS contrast (std of luminance)."),
                io.Combo.Input("if_none_pass", options=list(NONE_PASS_MODES), default="return_best",
                               tooltip="return_best: send the best-scoring image to 'passed' and flag it in the report. "
                                       "block: send nothing; nodes fed by 'passed' are skipped without error."),
            ],
            outputs=[
                io.Image.Output("passed"),
                io.Image.Output("rejected", tooltip="Skipped downstream (not an error) when nothing was rejected."),
                io.String.Output("report", tooltip="JSON report with metrics and reasons per image."),
                io.Int.Output("pass_count"),
            ],
        )

    @classmethod
    def validate_inputs(cls, min_brightness=0.08, max_brightness=0.92, **kwargs):
        if min_brightness >= max_brightness:
            return "min_brightness must be lower than max_brightness"
        return True

    @classmethod
    def execute(cls, images, min_sharpness, min_brightness, max_brightness,
                max_clipped_highlights, max_crushed_shadows, min_contrast, if_none_pass) -> io.NodeOutput:
        thresholds = Thresholds(
            min_sharpness=min_sharpness, min_mean_luma=min_brightness, max_mean_luma=max_brightness,
            max_clipped_highlights=max_clipped_highlights, max_crushed_shadows=max_crushed_shadows,
            min_contrast=min_contrast,
        )
        result = run_gate(images, thresholds, if_none_pass)
        report = result.to_dict(thresholds)

        # An empty IMAGE batch breaks many downstream nodes, so an empty branch is blocked
        # silently instead: whatever hangs off it simply does not run this time.
        passed = images[result.passed] if result.passed else ExecutionBlocker(None)
        rejected = images[result.rejected] if result.rejected else ExecutionBlocker(None)

        summary = f"{report['passed']}/{report['total']} passed"
        if result.fallback_used:
            summary += " - none met the thresholds, returning the best one (flagged in report)"
        for r in result.reports:
            if r.reasons:
                summary += f"\n#{r.index}: " + "; ".join(r.reasons)
        return io.NodeOutput(passed, rejected, json.dumps(report, indent=2), report["passed"],
                             ui=ui.PreviewText(summary))
