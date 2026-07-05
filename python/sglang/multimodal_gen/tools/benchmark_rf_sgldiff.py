#!/usr/bin/env python3
"""Benchmark SglDiff Rolling Forcing with paper-aligned streaming metrics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

PAPER_SAMPLE_FPS = 16
# Wan causal VAE temporal upsampling: 1 latent frame ~= 4 pixel frames in
# steady state (81 latent -> 321 pixel). The paper's "16 FPS" counts pixel
# frames at playback rate (sample_fps=16), so steady-state pixel FPS is the
# apples-to-apples metric.
PIXEL_FRAMES_PER_LATENT = 4
ROOT = Path(__file__).resolve().parents[1]
SGLANG_ROOT = Path(__file__).resolve().parents[4]


def compute_streaming_metrics(profile: dict, skip_ramp_windows: int = 5) -> dict:
    windows = profile["window_profiles"]
    fpb = profile["num_frame_per_block"]
    denoise_steps = profile["num_denoising_steps"]

    full_windows = [w for w in windows if w["is_full_window"]]
    steady_full = full_windows[skip_ramp_windows:]
    if not steady_full:
        steady_full = full_windows

    steady_no_tail = [
        w
        for w in steady_full
        if w["num_blocks"] == denoise_steps and w["start_block"] > 0
    ]
    if not steady_no_tail:
        steady_no_tail = steady_full

    def fps_from_windows(window_list, new_frames_per_window):
        if not window_list:
            return None
        median_ms = statistics.median(w["window_ms"] for w in window_list)
        return new_frames_per_window / (median_ms / 1000.0)

    diffusion_s = profile["diffusion_ms"] / 1000.0
    num_blocks = profile["num_blocks"]
    batch_dit_fps = (num_blocks * fpb) / diffusion_s if diffusion_s > 0 else None

    median_steady_ms = (
        statistics.median(w["window_ms"] for w in steady_no_tail)
        if steady_no_tail
        else None
    )

    return {
        "steady_windows_used": len(steady_no_tail),
        "median_steady_window_ms": median_steady_ms,
        "batch_dit_only_fps": batch_dit_fps,
        "streaming_block_fps": fps_from_windows(steady_no_tail, fpb),
        "streaming_latent_frame_fps": fps_from_windows(steady_no_tail, 1),
        "streaming_pixel_fps": fps_from_windows(
            steady_no_tail, fpb * PIXEL_FRAMES_PER_LATENT
        ),
    }


def print_metric_report(metrics: dict) -> None:
    print("\n=== SglDiff Rolling Forcing metrics ===")
    if metrics["batch_dit_only_fps"]:
        print(f"  [B] Batch DiT-only FPS: {metrics['batch_dit_only_fps']:.2f}")
    else:
        print("  [B] Batch DiT-only FPS: n/a")
    if metrics["streaming_block_fps"]:
        print(
            f"  [D] Streaming steady block FPS (median window, "
            f"{metrics['steady_windows_used']} windows): "
            f"{metrics['streaming_block_fps']:.2f}"
        )
    else:
        print("  [D] Streaming steady block FPS: n/a")
    if metrics["median_steady_window_ms"] is not None:
        print(f"      median steady window: {metrics['median_steady_window_ms']:.1f} ms")
    if metrics.get("streaming_pixel_fps"):
        print(
            f"  [D-pixel] Streaming steady PIXEL FPS "
            f"({3 * PIXEL_FRAMES_PER_LATENT} pixel frames / window): "
            f"{metrics['streaming_pixel_fps']:.2f}"
        )
    print(
        f"  Paper claim: ~{PAPER_SAMPLE_FPS} FPS = steady PIXEL frames at "
        f"playback rate (sample_fps=16); compare with [D-pixel]."
    )


def build_generate_cmd(args: argparse.Namespace, extra_flags: list[str]) -> list[str]:
    venv_sglang = SGLANG_ROOT / ".venv" / "bin" / "sglang"
    sglang_bin = str(venv_sglang if venv_sglang.is_file() else "sglang")
    cmd = [
        sglang_bin,
        "generate",
        "--model-path",
        args.model_path,
        "--pipeline-class-name",
        "WanRollingForcingPipeline",
        "--prompt",
        args.prompt,
        "--num-frames",
        str(args.num_frames),
        "--output-path",
        args.output_path,
        "--seed",
        str(args.seed),
        "--dit-cpu-offload",
        "false",
        "--height",
        str(args.height),
        "--width",
        str(args.width),
    ]
    cmd.extend(extra_flags)
    return cmd


def run_generate(
    args: argparse.Namespace, extra_flags: list[str], label: str, profile_path: Path
) -> float:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SGLANG_ROOT / "python")
    env["SGLANG_RF_PROFILE"] = "1"
    env["SGLANG_RF_PROFILE_PATH"] = str(profile_path)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cmd = build_generate_cmd(args, extra_flags)
    print(f"\n--- {label} ---")
    print(" ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(SGLANG_ROOT),
        capture_output=True,
        text=True,
    )
    wall_s = time.perf_counter() - start
    log = result.stdout + result.stderr
    print(log)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with code {result.returncode}")
    if profile_path.is_file():
        with profile_path.open(encoding="utf-8") as handle:
            profile = json.load(handle)
        print_metric_report(compute_streaming_metrics(profile))
    return wall_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/data/ckpts/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )
    parser.add_argument("--prompt", default="A cheetah running across the savanna")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--output-path",
        default="/data/osc7/outputs/rf_bench_sgldiff",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    baseline_flags = [
        "--text-encoder-cpu-offload",
        "--pin-cpu-memory",
    ]
    tuned_flags = [
        "--performance-mode",
        "speed",
    ]

    if not args.skip_baseline:
        baseline_s = run_generate(
            args,
            baseline_flags,
            "Baseline (text-encoder CPU offload)",
            Path(args.output_path) / "profile_baseline.json",
        )
    else:
        baseline_s = None

    tuned_s = run_generate(
        args,
        tuned_flags,
        "Tuned (performance-mode=speed, GPU-resident TE+VAE)",
        Path(args.output_path) / "profile_tuned.json",
    )

    latent_output_frames = args.num_frames
    print("\n=== Wall-clock e2e (includes VAE decode) ===")
    if baseline_s is not None:
        print(
            f"  Baseline: {baseline_s:.1f}s -> "
            f"{latent_output_frames / baseline_s:.2f} latent-FPS"
        )
    print(
        f"  Tuned:    {tuned_s:.1f}s -> "
        f"{latent_output_frames / tuned_s:.2f} latent-FPS"
    )
    if baseline_s is not None and tuned_s > 0:
        print(f"  Speedup:  {baseline_s / tuned_s:.2f}x")


if __name__ == "__main__":
    main()
