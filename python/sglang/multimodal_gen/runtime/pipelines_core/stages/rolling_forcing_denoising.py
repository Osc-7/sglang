# SPDX-License-Identifier: Apache-2.0
"""
Rolling Forcing denoising stage for Wan causal world models.

Ports TencentARC/RollingForcing ``inference_rolling_forcing`` to SglDiff's
B,C,T,H,W latent layout and causal KV cache infrastructure.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def compute_rolling_window_blocks(
    num_blocks: int, rolling_window_length_blocks: int
) -> tuple[list[int], list[int], int]:
    """Return start/end block indices for each rolling window."""
    window_num = num_blocks + rolling_window_length_blocks - 1
    window_start_blocks: list[int] = []
    window_end_blocks: list[int] = []
    for window_index in range(window_num):
        start_block = max(0, window_index - rolling_window_length_blocks + 1)
        end_block = min(num_blocks - 1, window_index)
        window_start_blocks.append(start_block)
        window_end_blocks.append(end_block)
    return window_start_blocks, window_end_blocks, window_num


def build_shared_timestep(
    denoising_step_list: torch.Tensor,
    batch_size: int,
    num_frame_per_block: int,
    rolling_window_length_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """Build per-frame progressive timesteps shared across full windows."""
    shared_timestep = torch.ones(
        [batch_size, rolling_window_length_blocks * num_frame_per_block],
        device=device,
        dtype=torch.float32,
    )
    for index, current_timestep in enumerate(reversed(denoising_step_list)):
        start = index * num_frame_per_block
        end = (index + 1) * num_frame_per_block
        shared_timestep[:, start:end] *= float(current_timestep)
    return shared_timestep


def slice_window_timestep(
    shared_timestep: torch.Tensor,
    *,
    current_num_frames: int,
    current_start_frame: int,
    num_frames: int,
    rolling_window_length_blocks: int,
    num_frame_per_block: int,
) -> torch.Tensor:
    full_window_frames = rolling_window_length_blocks * num_frame_per_block
    if current_num_frames == full_window_frames:
        return shared_timestep
    if current_start_frame == 0:
        return shared_timestep[:, -current_num_frames:]
    if current_start_frame + current_num_frames == num_frames:
        return shared_timestep[:, :current_num_frames]
    raise ValueError(
        "current_num_frames should equal rolling window length, or be the first/last window"
    )


class RollingForcingDenoisingStage(CausalDMDDenoisingStage):
    """Joint progressive denoising inside a rolling temporal window."""

    last_profile: dict | None = None

    def _should_profile(self) -> bool:
        return os.environ.get("SGLANG_RF_PROFILE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _predict_x0_per_frame_btchw(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        latent_model_input: torch.Tensor,
        noise_latents_btchw: torch.Tensor,
        timestep_bf: torch.Tensor,
        scheduler,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        attn_raw_latent_shape: tuple[int, int, int],
        current_timestep: int,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, object | None]:
        attn_metadata = self._build_causal_attn_metadata(
            batch,
            server_args,
            current_timestep=current_timestep,
            raw_latent_shape=attn_raw_latent_shape,
            device=device,
        )
        timestep = timestep_bf.to(
            device=latent_model_input.device, dtype=torch.long
        )
        pred_noise = self._forward_causal_transformer(
            batch,
            latent_model_input=latent_model_input,
            prompt_embeds=prompt_embeds,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            current_timestep=current_timestep,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )
        pred_noise_btchw = pred_noise.permute(0, 2, 1, 3, 4)
        x0_btchw = pred_noise_to_pred_video(
            pred_noise=pred_noise_btchw.flatten(0, 1),
            noise_input_latent=noise_latents_btchw.flatten(0, 1),
            timestep=timestep,
            scheduler=scheduler,
        ).unflatten(0, pred_noise_btchw.shape[:2])
        return x0_btchw, attn_metadata

    def _build_noisy_input(
        self,
        *,
        noisy_cache: torch.Tensor,
        noise: torch.Tensor,
        output_start_frame: int,
        output_end_frame: int,
        gen_start_frame: int,
        gen_end_frame: int,
        rolling_window_length_blocks: int,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        current_num_frames = output_end_frame - output_start_frame
        full_window_frames = rolling_window_length_blocks * num_frame_per_block
        if current_num_frames == full_window_frames or gen_start_frame == 0:
            return torch.cat(
                [
                    noisy_cache[
                        :,
                        :,
                        output_start_frame : output_end_frame - num_frame_per_block,
                    ],
                    noise[
                        :,
                        :,
                        gen_end_frame - num_frame_per_block : gen_end_frame,
                    ],
                ],
                dim=2,
            )
        return noisy_cache[:, :, output_start_frame:output_end_frame]

    def _update_noisy_cache(
        self,
        batch: Req,
        *,
        noisy_cache: torch.Tensor,
        denoised_pred: torch.Tensor,
        current_timestep: torch.Tensor,
        denoising_step_list: torch.Tensor,
        start_block: int,
        end_block: int,
        scheduler,
        device: torch.device,
        num_frame_per_block: int,
        current_num_frames: int,
        num_input_frames: int,
    ) -> None:
        batch_size = denoised_pred.shape[0]
        denoised_btchw = denoised_pred.permute(0, 2, 1, 3, 4)
        for block_idx in range(start_block, end_block + 1):
            block_offset = block_idx - start_block
            block_timestep = current_timestep[
                :,
                block_offset
                * num_frame_per_block : (block_offset + 1)
                * num_frame_per_block,
            ].mean()
            matches = torch.abs(denoising_step_list - block_timestep) < 1e-4
            block_timestep_index = torch.nonzero(matches, as_tuple=True)[0]
            if block_timestep_index.numel() == 0:
                continue
            if block_timestep_index.item() == len(denoising_step_list) - 1:
                continue
            next_timestep = denoising_step_list[block_timestep_index.item() + 1].to(
                device
            )
            noise = torch.randn(
                denoised_btchw.shape,
                dtype=denoised_btchw.dtype,
                generator=self._single_generator(batch),
                device=device,
            )
            renoised = scheduler.add_noise(
                denoised_btchw.flatten(0, 1),
                noise.flatten(0, 1),
                next_timestep
                * torch.ones(
                    [batch_size * current_num_frames],
                    device=device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_btchw.shape[:2])
            block_slice = slice(
                block_idx * num_frame_per_block + num_input_frames,
                (block_idx + 1) * num_frame_per_block + num_input_frames,
            )
            window_slice = slice(
                block_offset * num_frame_per_block,
                (block_offset + 1) * num_frame_per_block,
            )
            noisy_cache[:, :, block_slice] = renoised[:, window_slice].permute(
                0, 2, 1, 3, 4
            )

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        target_dtype = ctx.target_dtype
        autocast_enabled = ctx.autocast_enabled
        scheduler = ctx.scheduler
        device = ctx.device
        denoising_step_list = ctx.timesteps
        image_kwargs = ctx.image_kwargs
        pos_cond_kwargs = ctx.pos_cond_kwargs
        latents = ctx.latents
        noise = latents
        prompt_embeds = ctx.prompt_embeds
        batch_size, channels, num_frames, height, width = (
            ctx.batch_size,
            ctx.channels,
            ctx.num_frames,
            ctx.height,
            ctx.width,
        )

        independent_first_frame = self.transformer.independent_first_frame
        num_frame_per_block = self.num_frames_per_block
        num_denoising_steps = len(denoising_step_list)
        rolling_window_length_blocks = num_denoising_steps

        if self.causal_kv_cache is None:
            self._initialize_causal_caches(
                batch_size=batch_size,
                max_text_len=self._get_max_text_len(server_args),
                dtype=target_dtype,
                device=device,
            )
        else:
            assert self.crossattn_cache is not None
            self._reset_causal_caches(
                kv_cache=self.causal_kv_cache,
                crossattn_cache=self.crossattn_cache,
            )

        current_start_frame = 0
        if getattr(batch, "image_latent", None) is not None:
            image_latent = batch.image_latent
            assert image_latent is not None
            input_frames = image_latent.shape[2]
            if independent_first_frame and input_frames >= 1:
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[:, :, :1, :, :],
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += 1
                remaining_frames = input_frames - 1
            else:
                remaining_frames = input_frames

            while remaining_frames > 0:
                block = min(num_frame_per_block, remaining_frames)
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[
                        :, :, current_start_frame : current_start_frame + block, :, :
                    ],
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += block
                remaining_frames -= block

        num_input_frames = current_start_frame
        num_output_frames = num_frames + num_input_frames

        if not independent_first_frame or (
            independent_first_frame and batch.image_latent is not None
        ):
            if num_frames % num_frame_per_block != 0:
                raise ValueError(
                    "num_frames must be divisible by num_frames_per_block for Rolling Forcing"
                )
            num_blocks = num_frames // num_frame_per_block
        else:
            if (num_frames - 1) % num_frame_per_block != 0:
                raise ValueError(
                    "(num_frames - 1) must be divisible by num_frames_per_block when "
                    "independent_first_frame=True"
                )
            num_blocks = (num_frames - 1) // num_frame_per_block

        window_start_blocks, window_end_blocks, window_num = (
            compute_rolling_window_blocks(num_blocks, rolling_window_length_blocks)
        )
        shared_timestep = build_shared_timestep(
            denoising_step_list,
            batch_size,
            num_frame_per_block,
            rolling_window_length_blocks,
            device,
        )

        output = torch.zeros(
            [batch_size, channels, num_output_frames, height, width],
            device=device,
            dtype=noise.dtype,
        )
        if num_input_frames > 0:
            assert batch.image_latent is not None
            output[:, :, :num_input_frames] = batch.image_latent[
                :, :, :num_input_frames
            ]

        noisy_cache = torch.zeros_like(output)
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor] = lambda x: x

        profile = self._should_profile()
        window_profiles: list[dict] = []
        diffusion_start = diffusion_end = None
        if profile and device.type == "cuda":
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            diffusion_start.record()

        with self.progress_bar(total=window_num, batch=batch) as progress_bar:
            for window_index in range(window_num):
                win_ev_start = win_ev_end = None
                if profile and device.type == "cuda":
                    win_ev_start = torch.cuda.Event(enable_timing=True)
                    win_ev_end = torch.cuda.Event(enable_timing=True)
                    win_ev_start.record()

                start_block = window_start_blocks[window_index]
                end_block = window_end_blocks[window_index]
                gen_start_frame = start_block * num_frame_per_block
                gen_end_frame = (end_block + 1) * num_frame_per_block
                output_start_frame = gen_start_frame + num_input_frames
                output_end_frame = gen_end_frame + num_input_frames
                current_num_frames = output_end_frame - output_start_frame

                noisy_input = self._build_noisy_input(
                    noisy_cache=noisy_cache,
                    noise=noise,
                    output_start_frame=output_start_frame,
                    output_end_frame=output_end_frame,
                    gen_start_frame=gen_start_frame,
                    gen_end_frame=gen_end_frame,
                    rolling_window_length_blocks=rolling_window_length_blocks,
                    num_frame_per_block=num_frame_per_block,
                )
                current_timestep = slice_window_timestep(
                    shared_timestep,
                    current_num_frames=current_num_frames,
                    current_start_frame=gen_start_frame,
                    num_frames=num_frames,
                    rolling_window_length_blocks=rolling_window_length_blocks,
                    num_frame_per_block=num_frame_per_block,
                )

                latent_model_input = prepare_model_input(noisy_input).to(target_dtype)
                noise_latents_btchw = noisy_input.permute(0, 2, 1, 3, 4)
                denoised_pred, attn_metadata = self._predict_x0_per_frame_btchw(
                    batch,
                    server_args,
                    latent_model_input=latent_model_input,
                    noise_latents_btchw=noise_latents_btchw,
                    timestep_bf=current_timestep,
                    scheduler=scheduler,
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_tokens=output_start_frame * self.num_token_per_frame,
                    start_frame=output_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    attn_raw_latent_shape=(current_num_frames, height, width),
                    current_timestep=window_index,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                    device=device,
                )
                # _predict_x0_per_frame_btchw returns (B, T, C, H, W); the rest of
                # the RF stage (output, noisy_cache, context frames) uses (B, C, T, H, W).
                denoised_pred = denoised_pred.permute(0, 2, 1, 3, 4).contiguous()
                output[:, :, output_start_frame:output_end_frame] = denoised_pred

                self._update_noisy_cache(
                    batch,
                    noisy_cache=noisy_cache,
                    denoised_pred=denoised_pred,
                    current_timestep=current_timestep,
                    denoising_step_list=denoising_step_list,
                    start_block=start_block,
                    end_block=end_block,
                    scheduler=scheduler,
                    device=device,
                    num_frame_per_block=num_frame_per_block,
                    current_num_frames=current_num_frames,
                    num_input_frames=num_input_frames,
                )

                context_frames = denoised_pred[:, :, :num_frame_per_block]
                self._update_causal_context_cache(
                    batch,
                    server_args,
                    context_input=context_frames,
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_tokens=output_start_frame * self.num_token_per_frame,
                    start_frame=output_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    attn_metadata=attn_metadata,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                if profile and device.type == "cuda":
                    win_ev_end.record()
                    torch.cuda.synchronize()
                    window_ms = win_ev_start.elapsed_time(win_ev_end)
                    window_profiles.append(
                        {
                            "window_index": window_index,
                            "start_block": start_block,
                            "end_block": end_block,
                            "num_blocks": end_block - start_block + 1,
                            "num_frames": current_num_frames,
                            "is_full_window": current_num_frames
                            == rolling_window_length_blocks * num_frame_per_block,
                            "window_ms": window_ms,
                        }
                    )
                progress_bar.update()

        if profile and device.type == "cuda":
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_ms = diffusion_start.elapsed_time(diffusion_end)
            self.last_profile = {
                "diffusion_ms": diffusion_ms,
                "num_blocks": num_blocks,
                "num_denoising_steps": num_denoising_steps,
                "num_frame_per_block": num_frame_per_block,
                "window_profiles": window_profiles,
            }
            logger.info(
                "Rolling Forcing diffusion profile: %.1f ms over %d windows",
                diffusion_ms,
                len(window_profiles),
            )
            profile_path = os.environ.get("SGLANG_RF_PROFILE_PATH")
            if profile_path:
                import json

                with open(profile_path, "w", encoding="utf-8") as handle:
                    json.dump(self.last_profile, handle, indent=2)
                logger.info("Wrote RF profile to %s", profile_path)

        batch.latents = output
        return batch


from sglang.multimodal_gen.tools.wan_repack import TRANSFORMER_KEYS_RENAME_DICT


def _strip_native_wan_prefix(key: str) -> str:
    for prefix in ("model._fsdp_wrapped_module.", "model.", "_fsdp_wrapped_module."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def map_native_wan_state_dict_to_sglang(
    state_dict: dict[str, torch.Tensor],
    param_names_mapping: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Map TencentARC native Wan keys to SglDiff CausalWanTransformer3DModel keys.

    Two stages:
      1. native -> diffusers (via ``TRANSFORMER_KEYS_RENAME_DICT``)
      2. diffusers -> SglDiff internal (via the model's ``param_names_mapping``)
    """
    import re as _re

    if param_names_mapping is None:
        from sglang.multimodal_gen.configs.models.dits import WanVideoConfig

        param_names_mapping = WanVideoConfig().param_names_mapping

    mapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = _strip_native_wan_prefix(key)
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)
        for pattern, replacement in param_names_mapping.items():
            candidate = _re.sub(pattern, replacement, new_key)
            if candidate != new_key:
                new_key = candidate
                break
        if new_key:
            mapped[new_key] = value
    return mapped


def load_rolling_forcing_generator_checkpoint(
    transformer: torch.nn.Module,
    checkpoint_path: str,
    *,
    use_ema: bool = True,
) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = "generator_ema" if use_ema else "generator"
    if key not in checkpoint:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} missing '{key}' (available: {list(checkpoint)})"
        )
    param_names_mapping = getattr(
        getattr(transformer, "config", None), "param_names_mapping", None
    )
    state_dict = map_native_wan_state_dict_to_sglang(
        checkpoint[key], param_names_mapping=param_names_mapping
    )
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded Rolling Forcing checkpoint from %s (%s): missing=%d unexpected=%d",
        checkpoint_path,
        key,
        len(missing),
        len(unexpected),
    )
    if missing:
        logger.warning("Missing keys after RF load (first 30): %s", missing[:30])
    if unexpected:
        logger.warning("Unexpected keys after RF load (first 30): %s", unexpected[:30])
    return missing, unexpected
