# SPDX-License-Identifier: Apache-2.0
"""Helpers for batch generation output directories."""

from __future__ import annotations

import os
import re
import time

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_RUN_DIR_PATTERN = re.compile(r"^run_(\d+)_(\d{14})$")


def should_allocate_batch_run_output_dir(
    *,
    num_prompts: int,
    prompt_path: str | None,
    output_file_path_override: bool,
) -> bool:
    """Return True when outputs should go under ``run_<id>_<YYYYMMDDHHMMSS>/``."""
    if output_file_path_override:
        return False
    if num_prompts > 1:
        return True
    return bool(prompt_path)


def allocate_batch_run_output_dir(base_output_path: str | None) -> str:
    """Create ``{base}/run_<next_id>_<YYYYMMDDHHMMSS>/`` and return its path."""
    base = os.path.abspath(base_output_path or "outputs")
    os.makedirs(base, exist_ok=True)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    next_run_id = 0
    for name in os.listdir(base):
        match = _RUN_DIR_PATTERN.match(name)
        if match is not None:
            next_run_id = max(next_run_id, int(match.group(1)) + 1)

    run_dir = os.path.join(base, f"run_{next_run_id}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Batch outputs will be saved under %s", run_dir)
    return run_dir
