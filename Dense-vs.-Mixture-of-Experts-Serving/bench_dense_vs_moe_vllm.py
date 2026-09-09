#!/usr/bin/env python3
"""
Dense vs. Mixture-of-Experts serving throughput on a single GPU (vLLM).

Compares:
  - allenai/OLMo-2-0425-1B-Instruct        (dense, small)
  - allenai/OLMoE-1B-7B-0924-Instruct      (MoE, ~1B active / 7B total)
  - allenai/OLMo-2-1124-7B                 (dense, large)

served with FP8 weights + FP8 KV cache, across a prefill-dominated sweep
(vary input context length at fixed batch size) and a decode-dominated
sweep (vary number of parallel generations at fixed prompt/output length).

Usage:
    python bench_dense_vs_moe_vllm.py --model-key all
    python bench_dense_vs_moe_vllm.py --model-key moe_1b7b
"""
import argparse
import json
import os
import subprocess
import sys
import shutil
import time
import traceback
from pathlib import Path

# ------------------------------------------------------------
# Must be set BEFORE vllm is imported (vllm reads it at import time).
#
# FlashInfer's top-k/top-p sampling kernels are JIT-compiled on first
# use and require nvcc / a CUDA toolkit install at runtime. On machines
# with only the CUDA *runtime* (e.g. a pip-installed torch, no
# /usr/local/cuda), that JIT build fails with:
#     RuntimeError: Could not find nvcc and default
#     cuda_home='/usr/local/cuda' doesn't exist
# ...during the engine's startup profiling run, before any benchmark
# work happens.
#
# Disabling it makes vLLM use its native PyTorch/Triton sampler. This
# has no effect on our measurements: every benchmark request here uses
# temperature=0.0 (greedy), so the top-k/top-p sampler is never on the
# critical path. Only the engine's dummy profiling run touches it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODELS = {
    "dense_1b": {
        "hf_id": "allenai/OLMo-2-0425-1B-Instruct",
        "label": "OLMo 2 1B Dense",
        "architecture": "dense",
    },
    "moe_1b7b": {
        "hf_id": "allenai/OLMoE-1B-7B-0924-Instruct",
        "label": "OLMoE 1B active / 7B total",
        "architecture": "moe",
    },
    "dense_7b": {
        "hf_id": "allenai/OLMo-2-1124-7B",
        "label": "OLMo 2 7B Dense",
        "architecture": "dense",
    },
}

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
MAX_MODEL_LEN = 4096

# Prefill experiment:
# Keep batch size fixed; vary only context length.
PREFILL_LENGTHS = [128, 256, 512, 1024, 2048, 3072, 4095]
PREFILL_BATCH_SIZE = 4
PREFILL_OUTPUT_LEN = 1

# Decode experiment:
# Keep prompt/output lengths fixed; vary only parallel sequences.
DECODE_PROMPT_LEN = 32
DECODE_OUTPUT_LEN = 512
DECODE_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]

REPEATS = 3
GPU_MEMORY_UTILIZATION = 0.90

# FP8 *tensor-core* compute requires an Ada/Hopper-class GPU (compute
# capability >= 8.9). On Ampere (e.g. RTX 3090, cc 8.6) vLLM still
# accepts quantization="fp8", but falls back to WEIGHT-ONLY FP8 via
# Marlin kernels: weights are stored in FP8 and dequantized to BF16 for
# the matmul. You get the memory-footprint benefit but not FP8 math.
# This distinction matters for interpreting the prefill (compute-bound)
# results, so we detect it and record it in the run metadata.
MIN_FP8_COMPUTE_CAPABILITY = (8, 9)


def check_fp8_hardware_support():
    import torch

    if not torch.cuda.is_available():
        print("[WARN] No CUDA device visible; FP8 serving will fail.")
        return {"compute_capability": None, "fp8_native_compute": False}
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[INFO] GPU: {name} (compute capability {major}.{minor})")
    if (major, minor) < MIN_FP8_COMPUTE_CAPABILITY:
        print(
            "[NOTE] Compute capability is below "
            f"{MIN_FP8_COMPUTE_CAPABILITY[0]}.{MIN_FP8_COMPUTE_CAPABILITY[1]} "
            "(Ada/Hopper), so FP8 runs in WEIGHT-ONLY mode: weights are "
            "stored in FP8 and dequantized to BF16 for the matmul (Marlin "
            "kernels). Memory footprint benefits apply; FP8 tensor-core "
            "math does not. Note this when interpreting prefill results."
        )
    return {
        "compute_capability": f"{major}.{minor}",
        "fp8_native_compute": (major, minor) >= MIN_FP8_COMPUTE_CAPABILITY,
    }


def check_flashinfer_jit_support():
    """
    kv_cache_dtype="fp8" makes vLLM select FlashInfer's attention
    backend for FP8 decode/prefill (independent of the FlashInfer
    *sampler*, which VLLM_USE_FLASHINFER_SAMPLER=0 above disables).
    FlashInfer JIT-compiles that backend's CUDA kernels on first use,
    which requires nvcc -- a full CUDA *toolkit*, not just the CUDA
    *runtime* that ships with a pip-installed torch. Without nvcc this
    fails deep inside engine warmup with a long traceback ending in
    "Could not find nvcc and default cuda_home='/usr/local/cuda'
    doesn't exist". Since the assignment requires an FP8 KV cache, the
    fix is to install a toolkit (e.g. `conda install -c nvidia
    cuda-toolkit -y`) rather than switch attention backends, which
    would likely drop FP8 KV cache support entirely. We just check and
    warn early here so that's a one-line message, not a stack trace
    after minutes of model loading.
    """
    has_nvcc = shutil.which("nvcc") is not None
    has_cuda_home = bool(os.environ.get("CUDA_HOME")) or os.path.isdir(
        "/usr/local/cuda"
    )
    if not has_nvcc and not has_cuda_home:
        print(
            "[NOTE] No nvcc found (checked PATH, $CUDA_HOME, "
            "/usr/local/cuda). FlashInfer's FP8 attention backend "
            "JIT-compiles its kernels and needs nvcc; this run will "
            "likely fail during engine warmup. Fix: run "
            "`conda install -c nvidia cuda-toolkit -y` in this "
            "environment, then re-run. The first request after that "
            "will be slower (one-time JIT compile, then cached)."
        )


def make_exact_prompt(tokenizer, length):
    """
    Construct a valid prompt containing exactly `length` tokens.
    Using token IDs directly avoids small tokenizer-dependent changes
    in prompt length.
    """
    seed_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Language models process sequences of tokens. "
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("Tokenizer produced no tokens.")
    repetitions = (length // len(seed_ids)) + 1
    token_ids = (seed_ids * repetitions)[:length]
    assert len(token_ids) == length
    return token_ids


def make_prompts(token_ids, count, tokens_prompt_cls):
    """
    Wrap raw token-id lists in vLLM's TokensPrompt type. Passing bare
    List[int]/List[List[int]] objects as `prompts` relies on legacy,
    version-dependent input parsing in vLLM; TokensPrompt is the
    explicit, stable way to hand vLLM pre-tokenized input.
    """
    return [
        tokens_prompt_cls(prompt_token_ids=token_ids[:]) for _ in range(count)
    ]


def synchronize():
    """
    llm.generate() is blocking, but synchronize explicitly around timing
    for cleaner measurements.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_generate(llm, prompts, sampling_params):
    synchronize()
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    synchronize()
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def benchmark_one_model(model_key, output_dir):
    import pandas as pd
    import torch
    import transformers
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    try:
        from vllm import TokensPrompt
    except ImportError:
        from vllm.inputs import TokensPrompt

    config = MODELS[model_key]
    model_id = config["hf_id"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Benchmarking: {model_id}")
    print("=" * 80)

    hardware_info = check_fp8_hardware_support()
    check_flashinfer_jit_support()

    # trust_remote_code=True: OLMoE (and some OLMo revisions) ship custom
    # modeling/tokenizer code on the Hub. Without this flag, loading can
    # fail outright or silently fall back to an incompatible auto class,
    # depending on installed transformers/vllm versions.
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Same serving configuration for all three checkpoints.
    #
    # dtype="bfloat16":
    #     pin the compute dtype explicitly rather than relying on each
    #     checkpoint's default torch_dtype, so all three models are
    #     quantized from the same starting precision.
    #
    # quantization="fp8":
    #     online FP8 weight quantization
    #
    # kv_cache_dtype="fp8":
    #     FP8 KV cache
    #
    # trust_remote_code=True:
    #     required for OLMoE's custom architecture on some
    #     transformers/vllm version combinations.
    #
    # enable_prefix_caching=False:
    #     prevents benchmark repetitions from reusing previous prompt KV
    #     states and artificially increasing measured prefill throughput.
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        dtype="bfloat16",
        quantization="fp8",
        kv_cache_dtype="fp8",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=max(DECODE_CONCURRENCY),
        enable_prefix_caching=False,
        trust_remote_code=True,
        seed=0,
    )

    metadata = {
        "model_key": model_key,
        "model": model_id,
        "label": config["label"],
        "architecture": config["architecture"],
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": hardware_info["compute_capability"],
        # False => FP8 is weight-only (Marlin), compute happens in BF16.
        "fp8_native_compute": hardware_info["fp8_native_compute"],
        "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        "max_model_len": MAX_MODEL_LEN,
        "quantization": "fp8",
        "kv_cache_dtype": "fp8",
        "prefix_caching": False,
        "prefill_batch_size": PREFILL_BATCH_SIZE,
        "decode_prompt_len": DECODE_PROMPT_LEN,
        "decode_output_len": DECODE_OUTPUT_LEN,
        "repeats": REPEATS,
    }
    with open(output_dir / f"metadata_{model_key}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ============================================================
    # 1. Prefill-dominated benchmark
    # ============================================================
    prefill_rows = []
    sampling_prefill = SamplingParams(
        temperature=0.0,
        max_tokens=PREFILL_OUTPUT_LEN,
        ignore_eos=True,
    )
    for context_len in PREFILL_LENGTHS:
        print(f"\n[PREFILL] context={context_len}, batch={PREFILL_BATCH_SIZE}")
        token_ids = make_exact_prompt(tokenizer, context_len)
        prompts = make_prompts(token_ids, PREFILL_BATCH_SIZE, TokensPrompt)

        # One untimed warmup for this shape.
        llm.generate(prompts, sampling_prefill, use_tqdm=False)

        for repeat in range(REPEATS):
            outputs, elapsed = timed_generate(llm, prompts, sampling_prefill)
            input_tokens = context_len * PREFILL_BATCH_SIZE
            actual_output_tokens = sum(
                len(request_output.outputs[0].token_ids)
                for request_output in outputs
            )
            # Primary prefill metric:
            # aggregate number of input tokens processed per second.
            input_tokens_per_second = input_tokens / elapsed
            print(
                f"  repeat={repeat}: {input_tokens_per_second:.2f} input "
                f"tok/s ({elapsed:.4f} s)"
            )
            prefill_rows.append(
                {
                    "model_key": model_key,
                    "model": model_id,
                    "label": config["label"],
                    "architecture": config["architecture"],
                    "context_length": context_len,
                    "batch_size": PREFILL_BATCH_SIZE,
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "input_tokens": input_tokens,
                    "output_tokens": actual_output_tokens,
                    "tokens_per_second": input_tokens_per_second,
                }
            )

    prefill_df = pd.DataFrame(prefill_rows)
    prefill_file = output_dir / f"prefill_raw_{model_key}.csv"
    prefill_df.to_csv(prefill_file, index=False)
    print(f"\nSaved {prefill_file}")

    # ============================================================
    # 2. Decode-dominated benchmark
    # ============================================================
    decode_rows = []
    sampling_decode = SamplingParams(
        temperature=0.0,
        max_tokens=DECODE_OUTPUT_LEN,
        ignore_eos=True,
    )
    decode_token_ids = make_exact_prompt(tokenizer, DECODE_PROMPT_LEN)

    for concurrency in DECODE_CONCURRENCY:
        print(
            f"\n[DECODE] concurrent sequences={concurrency}, "
            f"prompt={DECODE_PROMPT_LEN}, output={DECODE_OUTPUT_LEN}"
        )
        prompts = make_prompts(decode_token_ids, concurrency, TokensPrompt)

        # Untimed warmup for this concurrency.
        llm.generate(prompts, sampling_decode, use_tqdm=False)

        for repeat in range(REPEATS):
            outputs, elapsed = timed_generate(llm, prompts, sampling_decode)
            generated_tokens = sum(
                len(request_output.outputs[0].token_ids)
                for request_output in outputs
            )
            output_tokens_per_second = generated_tokens / elapsed
            print(
                f"  repeat={repeat}: {output_tokens_per_second:.2f} output "
                f"tok/s ({elapsed:.4f} s)"
            )
            decode_rows.append(
                {
                    "model_key": model_key,
                    "model": model_id,
                    "label": config["label"],
                    "architecture": config["architecture"],
                    "parallel_generations": concurrency,
                    "prompt_length": DECODE_PROMPT_LEN,
                    "output_length": DECODE_OUTPUT_LEN,
                    "repeat": repeat,
                    "elapsed_s": elapsed,
                    "generated_tokens": generated_tokens,
                    "tokens_per_second": output_tokens_per_second,
                }
            )

    decode_df = pd.DataFrame(decode_rows)
    decode_file = output_dir / f"decode_raw_{model_key}.csv"
    decode_df.to_csv(decode_file, index=False)
    print(f"\nSaved {decode_file}")


def combine_and_plot(output_dir, model_keys=None):
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)
    model_keys = model_keys or list(MODELS.keys())

    prefill_frames = []
    decode_frames = []
    available_keys = []
    for model_key in model_keys:
        prefill_path = output_dir / f"prefill_raw_{model_key}.csv"
        decode_path = output_dir / f"decode_raw_{model_key}.csv"
        if not prefill_path.exists() or not decode_path.exists():
            print(
                f"[WARN] Skipping {model_key}: missing raw CSV(s) "
                f"(run for this model likely failed). Expected "
                f"{prefill_path} and {decode_path}."
            )
            continue
        prefill_frames.append(pd.read_csv(prefill_path))
        decode_frames.append(pd.read_csv(decode_path))
        available_keys.append(model_key)

    if not prefill_frames:
        raise RuntimeError(
            "No successful model runs found -- nothing to plot. Check "
            "per-model logs for errors."
        )

    prefill = pd.concat(prefill_frames, ignore_index=True)
    decode = pd.concat(decode_frames, ignore_index=True)
    prefill.to_csv(output_dir / "prefill_raw_all.csv", index=False)
    decode.to_csv(output_dir / "decode_raw_all.csv", index=False)

    # ------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------
    prefill_summary = (
        prefill.groupby(
            ["model_key", "label", "architecture", "context_length"]
        )["tokens_per_second"]
        .agg(["mean", "std"])
        .reset_index()
    )
    decode_summary = (
        decode.groupby(
            ["model_key", "label", "architecture", "parallel_generations"]
        )["tokens_per_second"]
        .agg(["mean", "std"])
        .reset_index()
    )
    prefill_summary.to_csv(output_dir / "prefill_summary.csv", index=False)
    decode_summary.to_csv(output_dir / "decode_summary.csv", index=False)

    # ------------------------------------------------------------
    # Figure 1: Prefill
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    for model_key in available_keys:
        config = MODELS[model_key]
        df = prefill_summary[
            prefill_summary["model_key"] == model_key
        ].sort_values("context_length")
        plt.errorbar(
            df["context_length"],
            df["mean"],
            yerr=df["std"],
            marker="o",
            capsize=3,
            label=config["label"],
        )
    plt.xlabel("Input context length (tokens)")
    plt.ylabel("Average prefill throughput (input tokens/s)")
    plt.title("Prefill-Dominated Throughput")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "prefill_throughput.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # Figure 2: Decode
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 4.5))
    for model_key in available_keys:
        config = MODELS[model_key]
        df = decode_summary[
            decode_summary["model_key"] == model_key
        ].sort_values("parallel_generations")
        plt.errorbar(
            df["parallel_generations"],
            df["mean"],
            yerr=df["std"],
            marker="o",
            capsize=3,
            label=config["label"],
        )
    plt.xlabel("Number of parallel generations")
    plt.ylabel("Average decode throughput (output tokens/s)")
    plt.title("Decode-Dominated Throughput")
    # Concurrency values are powers of two.
    plt.xscale("log", base=2)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "decode_throughput.png", dpi=200)
    plt.close()

    print("\nCombined results:")
    print(output_dir / "prefill_raw_all.csv")
    print(output_dir / "decode_raw_all.csv")
    print(output_dir / "prefill_summary.csv")
    print(output_dir / "decode_summary.csv")
    print(output_dir / "prefill_throughput.png")
    print(output_dir / "decode_throughput.png")
    if len(available_keys) < len(model_keys):
        missing = sorted(set(model_keys) - set(available_keys))
        print(f"[WARN] Figures exclude models with no data: {missing}")


def run_all_models(args):
    """
    Run each model in a fresh Python process.
    This is intentional: destroying a vLLM object inside one Python
    process does not always release every CUDA allocation cleanly.
    A new process gives each model an equivalent fresh-GPU state.

    Each model's subprocess is isolated with try/except so that one
    model failing (e.g. unsupported FP8 quantization path for a given
    architecture, or an OOM) does not discard the other models' data --
    the run log records which models succeeded.
    """
    script = os.path.abspath(__file__)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log = {}
    for model_key in MODELS:
        command = [
            sys.executable,
            script,
            "--model-key",
            model_key,
            "--output-dir",
            args.output_dir,
        ]
        print("\n" + "#" * 80)
        print(" ".join(command))
        print("#" * 80 + "\n")
        try:
            subprocess.run(command, check=True)
            run_log[model_key] = {"status": "ok"}
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] {model_key} failed with exit code {exc.returncode}.")
            print(
                "        Continuing with remaining models; see stdout above "
                "for the failing model's traceback."
            )
            run_log[model_key] = {
                "status": "failed",
                "returncode": exc.returncode,
            }

    with open(output_dir / "run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    succeeded = [k for k, v in run_log.items() if v["status"] == "ok"]
    if not succeeded:
        raise RuntimeError("All model runs failed; see run_log.json and logs above.")

    combine_and_plot(args.output_dir, model_keys=succeeded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        choices=["all"] + list(MODELS.keys()),
        default="all",
    )
    parser.add_argument("--output-dir", default="results/moe_serving")
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help=(
            "Skip benchmarking; just rebuild the summary tables and both "
            "figures from raw CSVs already in --output-dir. Use this after "
            "running models individually with --model-key. Models with no "
            "raw CSVs are skipped with a warning."
        ),
    )
    args = parser.parse_args()

    if args.combine_only:
        combine_and_plot(args.output_dir)
    elif args.model_key == "all":
        run_all_models(args)
    else:
        try:
            benchmark_one_model(args.model_key, args.output_dir)
        except Exception:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
