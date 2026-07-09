# SPDX-License-Identifier: Apache-2.0
"""
Wan Rolling Forcing pipeline for long-video causal world-model inference.
"""

from __future__ import annotations

import json
import os
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    RollingForcingWanT2V480PConfig,
)
from sglang.multimodal_gen.configs.sample.wan import (
    RollingForcingWanT2V480PSamplingParams,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    RollingForcingDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.rolling_forcing_denoising import (
    load_rolling_forcing_generator_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_RF_CHECKPOINT = (
    "/data/ckpts/TencentARC/RollingForcing/checkpoints/rolling_forcing_dmd.pt"
)


class WanRollingForcingPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "WanRollingForcingPipeline"
    pipeline_config_cls = RollingForcingWanT2V480PConfig
    sampling_params_cls = RollingForcingWanT2V480PSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    _orig_transformer_config: dict[str, Any] | None = None

    def _patch_transformer_config(self, server_args: ServerArgs) -> str | None:
        transformer_path = os.path.join(self.model_path, "transformer")
        config_path = os.path.join(transformer_path, "config.json")
        if not os.path.isfile(config_path):
            logger.warning(
                "Transformer config not found at %s; skipping causal patch", config_path
            )
            return None

        with open(config_path, encoding="utf-8") as handle:
            self._orig_transformer_config = json.load(handle)

        arch = server_args.pipeline_config.dit_config.arch_config
        patched = dict(self._orig_transformer_config)
        patched["_class_name"] = "CausalWanTransformer3DModel"
        patched["local_attn_size"] = arch.local_attn_size
        patched["sink_size"] = arch.sink_size
        patched["num_frames_per_block"] = arch.num_frames_per_block
        patched["sliding_window_num_frames"] = arch.sliding_window_num_frames

        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(patched, handle, indent=2)
        return config_path

    def _restore_transformer_config(self, config_path: str | None) -> None:
        if config_path is None or self._orig_transformer_config is None:
            return
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(self._orig_transformer_config, handle, indent=2)
        self._orig_transformer_config = None

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict | None = None,
    ) -> dict:
        config_path = self._patch_transformer_config(server_args)
        try:
            return super().load_modules(server_args, loaded_modules)
        finally:
            self._restore_transformer_config(config_path)

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        pcfg = server_args.pipeline_config
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=pcfg.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

        # CLI `--rolling-forcing-checkpoint-path X` lands in
        # server_args.component_paths["rolling_forcing_checkpoint"] via the
        # generic --<component>-path extractor; it takes priority over the
        # pipeline config default.
        checkpoint_path = server_args.component_paths.get(
            "rolling_forcing_checkpoint"
        )
        if checkpoint_path is None:
            checkpoint_path = getattr(
                pcfg, "rolling_forcing_checkpoint_path", _DEFAULT_RF_CHECKPOINT
            )
        use_ema = getattr(pcfg, "rolling_forcing_use_ema", True)
        if not checkpoint_path:
            # Explicit opt-out (empty path): run with base Wan weights.
            logger.warning(
                "Rolling Forcing checkpoint disabled; using base transformer weights"
            )
            return
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Rolling Forcing checkpoint not found at {checkpoint_path!r}. "
                "Pass --rolling-forcing-checkpoint-path /path/to/rolling_forcing_dmd.pt "
                "(or set rolling_forcing_checkpoint_path in the pipeline config). "
                "To intentionally run with base Wan weights, pass "
                "--rolling-forcing-checkpoint-path ''."
            )
        logger.info("Loading Rolling Forcing checkpoint from %s", checkpoint_path)
        load_rolling_forcing_generator_checkpoint(
            self.get_module("transformer"),
            checkpoint_path,
            use_ema=use_ema,
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()
        self.add_stage(
            RollingForcingDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = WanRollingForcingPipeline
