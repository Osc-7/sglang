# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

from sglang.multimodal_gen.runtime.entrypoints.output_run_paths import (
    allocate_batch_run_output_dir,
    should_allocate_batch_run_output_dir,
)


def test_should_allocate_batch_run_output_dir():
    assert should_allocate_batch_run_output_dir(
        num_prompts=3,
        prompt_path=None,
        output_file_path_override=False,
    )
    assert should_allocate_batch_run_output_dir(
        num_prompts=1,
        prompt_path="/tmp/prompts.txt",
        output_file_path_override=False,
    )
    assert not should_allocate_batch_run_output_dir(
        num_prompts=1,
        prompt_path=None,
        output_file_path_override=False,
    )
    assert not should_allocate_batch_run_output_dir(
        num_prompts=3,
        prompt_path=None,
        output_file_path_override=True,
    )


def test_allocate_batch_run_output_dir_increments_run_id():
    with tempfile.TemporaryDirectory() as base:
        first = allocate_batch_run_output_dir(base)
        second = allocate_batch_run_output_dir(base)
        assert os.path.basename(first).startswith("run_0_")
        assert os.path.basename(second).startswith("run_1_")
        assert os.path.isdir(first)
        assert os.path.isdir(second)
