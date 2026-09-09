# Dense vs. Mixture-of-Experts Serving Throughput (vLLM, FP8)

Single-GPU vLLM benchmark comparing a dense and a Mixture-of-Experts (MoE)
checkpoint to find when MoE routing actually buys serving throughput over a
dense model of comparable size — and when it doesn't.

**Script:** [`bench_dense_vs_moe_vllm.py`](./bench_dense_vs_moe_vllm.py)

## Checkpoints

| Checkpoint | Architecture | Role |
|---|---|---|
| `allenai/OLMo-2-0425-1B-Instruct` | Dense | small dense baseline |
| `allenai/OLMoE-1B-7B-0924-Instruct` | MoE | ~1B active / 7B total params |
| `allenai/OLMo-2-1124-7B` | Dense | large dense baseline |

All three are served with **FP8 weights** and an **FP8 KV cache**, with the
rest of the serving configuration held fixed (`max_model_len`, GPU memory
fraction, seed, prefix caching disabled) so the only things that differ
across runs are the checkpoints themselves.

## Method

The script runs two independent sweeps per model, each isolating one
regime:

- **Prefill-dominated.** Batch size fixed at 4, input context length swept
  over `[128, 256, 512, 1024, 2048, 3072, 4095]` tokens, output capped at 1
  token (`ignore_eos=True`). Metric: aggregate input tokens/sec. This
  isolates how throughput scales with context length and with the
  per-token compute cost of each architecture.
- **Decode-dominated.** Prompt fixed at 32 tokens, output fixed at 512
  tokens, concurrency swept over `[1, 2, 4, 8, 16, 32, 64]` parallel
  sequences. Metric: aggregate output tokens/sec. This isolates how
  throughput scales with batching, which is where memory bandwidth (not
  compute) is usually the bottleneck.

Each configuration runs 1 untimed warmup + 3 timed repeats. Prompts are
built from exact token-ID sequences (not text) so context lengths are
precise regardless of each model's tokenizer. Every model runs in its own
subprocess so GPU memory from one model can't leak into the next.

## Requirements

- Single NVIDIA GPU, CUDA driver installed.
- A CUDA **toolkit** (`nvcc`) in the environment — not just the CUDA
  runtime that ships with pip-installed `torch`. vLLM's FP8 KV-cache
  attention path (FlashInfer) JIT-compiles kernels on first use and needs
  a real compiler. See [Troubleshooting](#troubleshooting) if you hit
  `Could not find nvcc`.
- Python 3.10+, in a **dedicated virtual environment**. vLLM pins
  aggressive `torch`/`transformers` versions; don't install it into an
  environment you rely on for other projects.

```bash
conda create -n moebench python=3.12 -y
conda activate moebench
pip install vllm pandas matplotlib
```

### Hardware note: FP8 compute vs. FP8 storage

FP8 *tensor-core* matmuls require an Ada/Hopper-class GPU (compute
capability ≥ 8.9, e.g. L4, L40, H100). On Ampere (e.g. RTX 3090/A100,
compute capability 8.6), vLLM still accepts `quantization="fp8"` but falls
back to **weight-only** FP8: weights are stored in FP8 (halving memory
footprint) and dequantized to BF16 for the actual matmul. You get the
memory benefit, not the compute speedup. The script detects this
automatically and records it in each model's `metadata_*.json`
(`fp8_native_compute: true/false`) — check this before interpreting the
prefill (compute-bound) numbers, since it changes what the prefill curve
is actually measuring.

## Usage

Run everything (recommended — downloads and benchmarks all three models
in sequence, then produces both figures):

```bash
python bench_dense_vs_moe_vllm.py --model-key all --output-dir results/moe_serving
```

Run one model at a time (useful on disk- or memory-constrained machines,
or to retry a single failed model):

```bash
python bench_dense_vs_moe_vllm.py --model-key dense_1b  --output-dir results/moe_serving
python bench_dense_vs_moe_vllm.py --model-key moe_1b7b  --output-dir results/moe_serving
python bench_dense_vs_moe_vllm.py --model-key dense_7b  --output-dir results/moe_serving
```

Rebuild the summary tables and figures from existing raw CSVs without
re-running anything (e.g. after running models individually):

```bash
python bench_dense_vs_moe_vllm.py --combine-only --output-dir results/moe_serving
```

A model with no raw CSVs in `--output-dir` is skipped with a warning
rather than aborting the whole combine step.

## Output

```
results/moe_serving/
├── metadata_<model_key>.json     # env, GPU, FP8 mode, config per model
├── prefill_raw_<model_key>.csv   # every (context_length, repeat) measurement
├── decode_raw_<model_key>.csv    # every (concurrency, repeat) measurement
├── prefill_raw_all.csv           # concatenated across models
├── decode_raw_all.csv
├── prefill_summary.csv           # mean ± std per (model, context_length)
├── decode_summary.csv            # mean ± std per (model, concurrency)
├── prefill_throughput.png        # Figure 1
└── decode_throughput.png         # Figure 2
└── run_log.json                  # per-model success/failure (--model-key all only)
```

## Interpreting the results

The two sweeps are designed to separate two different resource
bottlenecks, and the MoE checkpoint is expected to track a different
dense baseline in each:

- **Prefill is compute-bound.** FLOPs per token scale with *active*
  parameters, not total parameters. Since OLMoE activates ~1B parameters
  per token — about the same as the 1B dense model — its prefill
  throughput should track the **small dense baseline**, not the 7B, even
  though its total parameter count is 7B. (On an Ampere GPU where FP8 is
  weight-only, this comparison happens in BF16 compute for all three
  models, which is still a fair comparison of compute cost, just not of
  FP8 tensor-core throughput specifically.)
- **Decode at higher concurrency is memory-bandwidth-bound.** Every
  decode step has to stream each active parameter's weights from HBM. At
  concurrency 1, only the experts selected by that single token's routing
  decision need to be touched — MoE looks cheap, like the small dense
  model. As concurrency grows, different sequences in the batch route to
  different experts, and the union of experts touched per step grows
  toward the full expert set. At high concurrency, OLMoE's decode
  throughput should degrade toward — or converge with — the **large
  dense baseline**, because the effective memory traffic per step
  approaches that of a model with all 7B parameters resident and active.

This is the core trade a MoE is making: dense-active-parameter compute
cost, but large-total-parameter memory footprint. The crossover
concurrency where OLMoE stops looking like the 1B model and starts
looking like the 7B model is the practical answer to "when does MoE
routing pay off" — read it off `decode_throughput.png` and
`decode_summary.csv`. See Muennighoff et al., *OLMoE: Open Mixture-of-
Experts Language Models* (2025), for the architecture's expert-routing
design and the authors' own throughput/FLOPs analysis.

## Troubleshooting

Issues actually hit while developing this benchmark, in case they recur
on another machine:

- **`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't
  exist`** — FlashInfer's FP8 attention kernels need a CUDA toolkit, not
  just a runtime. Install one into your env: `conda install -c nvidia
  cuda-toolkit -y`.
- **`ld: cannot find -lcuda`** — after installing `cuda-toolkit`, the
  linker needs the driver stub library on its search path. Symlink it in:
  ```bash
  mkdir -p $CONDA_PREFIX/lib64/stubs
  ln -sf $CONDA_PREFIX/lib/stubs/libcuda.so $CONDA_PREFIX/lib64/stubs/libcuda.so
  ```
- **FlashInfer JIT errors during the *sampler* specifically** (separate
  from the attention backend above) — the script sets
  `VLLM_USE_FLASHINFER_SAMPLER=0` before importing vLLM. Every request
  here uses greedy decoding (`temperature=0.0`), so this has no effect on
  the measurements.
- **`OSError: No space left on device` / Xet `Background writer channel
  closed`** — both are disk-full symptoms, not vLLM bugs. FP8 doesn't
  shrink the *download* (checkpoints are fetched in their original
  precision and quantized on load), so budget ~3GB / ~14GB / ~14GB of
  disk for the three checkpoints respectively, on top of the ~10–15GB the
  `vllm` environment itself takes. On a disk you can't free up, point
  `HF_HOME` at a RAM-backed filesystem instead: `export
  HF_HOME=/dev/shm/$USER-hf` (clean it up after — it's RAM, not disk).
- **GPU OOM at high concurrency on the 7B model** — lower the top end of
  `DECODE_CONCURRENCY` in the script rather than `MAX_MODEL_LEN`; the
  prefill sweep needs the full context length.
