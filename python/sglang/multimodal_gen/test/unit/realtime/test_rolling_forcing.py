# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.rolling_forcing_denoising import (
    build_shared_timestep,
    compute_rolling_window_blocks,
    slice_window_timestep,
)


def test_compute_rolling_window_blocks_matches_reference():
    num_blocks = 7
    rolling_window_length_blocks = 5
    starts, ends, window_num = compute_rolling_window_blocks(
        num_blocks, rolling_window_length_blocks
    )
    assert window_num == num_blocks + rolling_window_length_blocks - 1
    assert len(starts) == window_num
    assert len(ends) == window_num

    expected_starts = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    expected_ends = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6]
    assert starts == expected_starts
    assert ends == expected_ends


def test_build_shared_timestep_progressive_noise():
    denoising_step_list = torch.tensor([1000, 800, 600, 400, 200])
    shared = build_shared_timestep(
        denoising_step_list,
        batch_size=2,
        num_frame_per_block=3,
        rolling_window_length_blocks=5,
        device=torch.device("cpu"),
    )
    assert shared.shape == (2, 15)
    assert torch.all(shared[:, 0:3] == 200)
    assert torch.all(shared[:, 3:6] == 400)
    assert torch.all(shared[:, 12:15] == 1000)


def test_slice_window_timestep_edges():
    shared = build_shared_timestep(
        torch.tensor([1000, 800, 600, 400, 200]),
        batch_size=1,
        num_frame_per_block=3,
        rolling_window_length_blocks=5,
        device=torch.device("cpu"),
    )
    full = slice_window_timestep(
        shared,
        current_num_frames=15,
        current_start_frame=0,
        num_frames=21,
        rolling_window_length_blocks=5,
        num_frame_per_block=3,
    )
    assert torch.equal(full, shared)

    tail = slice_window_timestep(
        shared,
        current_num_frames=9,
        current_start_frame=12,
        num_frames=21,
        rolling_window_length_blocks=5,
        num_frame_per_block=3,
    )
    assert tail.shape == (1, 9)
    assert torch.equal(tail, shared[:, :9])


def test_rolling_forcing_config_uses_latent_frame_count():
    from sglang.multimodal_gen.configs.pipeline_configs.wan import (
        RollingForcingWanT2V480PConfig,
        WanT2V480PConfig,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
        LatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    rf_config = RollingForcingWanT2V480PConfig()
    assert rf_config.vae_config.use_temporal_scaling_frames is False

    vanilla_config = WanT2V480PConfig()
    assert vanilla_config.vae_config.use_temporal_scaling_frames is True

    server_args = ServerArgs(
        model_path="/data/ckpts/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        pipeline_config=rf_config,
    )
    batch = Req(prompt="test", num_frames=81, height=480, width=832)
    latent_frames = LatentPreparationStage.adjust_video_length(
        LatentPreparationStage.__new__(LatentPreparationStage),
        batch,
        server_args,
    )
    assert latent_frames == 81


def test_self_forcing_deployment_config_keeps_vae_and_text_encoder_resident():
    from sglang.multimodal_gen.configs.pipeline_configs.wan import (
        RollingForcingWanT2V480PConfig,
        SelfForcingWanT2V480PConfig,
    )

    for config_cls in (SelfForcingWanT2V480PConfig, RollingForcingWanT2V480PConfig):
        deployment = config_cls().get_model_deployment_config()
        assert deployment.keep_resident_min_available_gb == 60
        assert "text_encoder" in deployment.keep_resident_components
        assert "vae" in deployment.keep_resident_components
