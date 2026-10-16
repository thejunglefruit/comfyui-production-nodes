# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 The Jungle Fruit
"""ComfyUI V3 node wrappers. Importing this package requires a running ComfyUI environment."""

from .apply_lut import ApplyCubeLUT, register_lut_folders
from .prompt_batch import PromptBatchCSV
from .quality_gate import QualityGate
from .run_manifest import RunManifest

NODES = [PromptBatchCSV, QualityGate, ApplyCubeLUT, RunManifest]

__all__ = ["NODES", "register_lut_folders", "PromptBatchCSV", "QualityGate", "ApplyCubeLUT", "RunManifest"]
