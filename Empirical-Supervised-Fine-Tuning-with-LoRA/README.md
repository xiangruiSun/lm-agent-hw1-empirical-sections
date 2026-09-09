# Supervised Fine-Tuning of Qwen2.5-0.5B with LoRA

An SFT pipeline for `Qwen/Qwen2.5-0.5B` on `HuggingFaceH4/ultrachat_200k`, comparing
LoRA at ranks {1, 4, 16} against full-weight fine-tuning under an otherwise identical
training setup.

Headline result: at this data scale LoRA rank is not the binding variable — ranks 1, 4
and 16 land within 0.005 nats of each other — while the LoRA-vs-full distinction is
decisive. Full fine-tuning at the same learning rate overfits 972 conversations badly
enough to end up *worse than the untrained model* on held-out data.

## Layout

| File | Role |
|---|---|
| `data.py` | UltraChat loading, chat-template formatting, **assistant-only loss masking**, padding collator |
| `train.py` | One training run (LoRA at a given rank, or full fine-tuning); records params / memory / time / loss curves to `metrics.json` |
| `generate.py` | Qualitative base-vs-fine-tuned generations on held-out prompts |
| `report.py` | Loss-curve figure + summary table across runs |
| `run_all.sh` | All five configurations, then the report and generations |

## Quickstart

```bash
conda create -n sft python=3.11 -y && conda activate sft
pip install torch --index-url https://download.pytorch.org/whl/cu124   # install first, on its own
pip install -r requirements.txt
bash run_all.sh
```

Install torch separately from the CUDA index. If you let `requirements.txt` pull it in as
a transitive dependency you can land on a CPU wheel, and the failure mode is a run that
completes but takes twenty hours.

Outputs land in `runs/<config>/metrics.json` and `results/`
(`loss_curves.png`, `summary.md`, `generations.md`). The full sweep is ~25 minutes on one
RTX 3090; budget ~4 GB of disk for the model, dataset shard and the fp32 full-tuning
checkpoints.

Single run:

```bash
python train.py --mode lora --rank 4 --out runs/lora_r4 \
  --n-train 1000 --n-val 100 --max-len 1024 --epochs 3 \
  --batch-size 2 --grad-accum 8 --lr 1e-4 --seed 0
```

Add `--streaming` to read the dataset over the wire instead of caching it, if disk is
tight. It draws a different (still deterministic) 1000-conversation sample, so pick one
mode and use it for the whole sweep.

## Method

**Loss is computed on assistant tokens only.** Each conversation is rendered with Qwen's
chat template; user turns, the system turn and all `<|im_start|>role` headers are masked
to `-100`. Each assistant turn's closing `<|im_end|>` *is* supervised — that is what
teaches the model to stop. Chat templates are prefix-consistent, so the assistant span is
recovered exactly as the difference between
`apply_chat_template(messages[:i], add_generation_prompt=True)` and
`apply_chat_template(messages[:i+1])`; `data.py` asserts this rather than assuming it.

**Everything except the parameterization is held fixed.** Same fixed seed-0 sample of
1000 `train_sft` conversations, same 100 held-out `test_sft` conversations, 3 epochs,
effective batch 16 (2 × 8 accumulation), lr 1e-4 with 3% warmup and cosine decay,
max length 1024, bf16 autocast, seed 0, same evaluation schedule.

**LoRA `alpha = 2r`.** This holds the scaling factor `alpha/r` constant across ranks, so a
rank difference isn't confounded with an effective learning-rate difference on the
adapter. Override with `--lora-alpha`.

**Validation loss is token-weighted.** Per-batch means are re-weighted by supervised token
count, giving a true average over the held-out set rather than an average of per-batch
averages.

**Memory is measured** with `max_memory_allocated()` after `reset_peak_memory_stats()`,
immediately before the training loop, with weights in fp32 under bf16 autocast in every
run.

## Results

RTX 3090 (24 GB) · 972 train / 98 val conversations · 3 epochs · effective batch 16 ·
max_len 1024.

| Configuration | Trainable params | % of total | Peak GPU mem (GiB) | Time (s) | s/step | Val loss | Val PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA r=1 | 549,888 | 0.111% | 9.31 | 298 | 1.65 | 1.3740 | 3.95 |
| LoRA r=4 | 2,199,552 | 0.443% | 9.34 | 306 | 1.70 | 1.3697 | 3.93 |
| LoRA r=16 | 8,798,208 | 1.750% | 9.46 | 305 | 1.69 | 1.3705 | 3.94 |
| Full, lr 1e-4 | 494,032,768 | 100.000% | 15.13 | 313 | 1.74 | 1.8902 | 6.62 |
| Full, lr 1e-5 | 494,032,768 | 100.000% | 15.13 | 312 | 1.73 | 1.3962 | 4.04 |
| *base model (no SFT)* | 0 | 0% | — | — | — | 1.4170 | 4.12 |

![loss curves](results/loss_curves.png)

The lr 1e-4 full-tuning row is the learning-rate-matched comparison; the lr 1e-5 row is
supplementary, included so it is clear full tuning is not merely handicapped by being
forced onto LoRA's learning rate.

### What the numbers say

**Rank buys almost nothing here.** Validation loss across r ∈ {1, 4, 16} spans 0.004
nats, which is noise. Even r=1's 550K parameters — 0.11% of the model — has enough
capacity for 972 conversations. Rank would start to matter with substantially more data.

**LoRA's saving is optimizer state, not activations.** The 5.8 GiB gap between LoRA and
full tuning matches the 5.5 GiB predicted for fp32 gradients plus AdamW's two moments over
494M parameters. The shared ~9.3 GiB floor is activations — with a 151,936-token
vocabulary the logits tensor is the single largest allocation — which is why peak memory
is flat across rank (9.31 → 9.46 GiB from r=1 to r=16).

**LoRA is not faster.** 298s vs 313s, about 5%. The backward pass still propagates through
every frozen layer; only the weight-gradient computation and the optimizer update are
skipped. LoRA saves memory and checkpoint size, not compute.

**Learning rate is the operative variable, not rank.** At lr 1e-4 full tuning's training
loss collapses to 0.40 with visible step-changes at the epoch boundaries (steps ~62 and
~124 — memorization), while validation loss climbs monotonically to 1.890, worse than the
1.417 of the untrained model. Dropping to 1e-5 removes the divergence (train 0.94, val
1.396) but still doesn't beat LoRA. The low-rank constraint acts as an implicit
regularizer that tolerates a learning rate full tuning cannot survive.

### Qualitative

Greedy decoding, 256 new tokens, identical prompt and chat template for every model. Full
transcripts in `results/generations.md`.

- **Base** — fluent continuation with no assistant persona, degenerating into verbatim
  loops ("I have been going through a lot of self-doubt and self-loathing…" four times).
- **LoRA, all ranks** — acquires the conversational surface (subject lines, numbered
  lists, "I'm sorry, but I cannot…") but leaks training artifacts and often fails to
  terminate: r=1 ends a landlord email with `[Your Name] Kylie / Geschäftez / You are a
  helpful assistant`; r=4 emits `LIBINT :NSUTF` and then hallucinates a further user turn.
  Differences between ranks are not systematic.
- **Full, lr 1e-4** — the largest register shift (rewrites the hash-table answer as a
  beginner analogy) but loses the role: its "email to my landlord" is addressed *to the
  tenant* and advises them to "contact your landlord". Consistent with its degraded
  validation loss.
- **Full, lr 1e-5** — the best instruction-follower. The only model that answers the
  grilled-chicken prompt with real numbered steps, times, temperatures and safety tips,
  and the only one to emit a well-formed *subsequent* turn ("That's really interesting!
  Can you give me an example…?" followed by an answer) — it learned UltraChat's multi-turn
  format but under-learned the turn-ending token.

Note that `Qwen2.5-0.5B` is the *base* model, not Instruct, so the base-vs-tuned contrast
is a real register shift. It is not a clean win, though: no configuration reliably emits
the stop token, and fine-tuning mostly changes *which* degeneration mode you get.

## Notes and gotchas

**`--lr` must not appear twice.** `argparse` keeps the last occurrence of a repeated flag.
An earlier version of `run_all.sh` had `--lr 1e-4` inside `$COMMON`, which expanded *after*
the explicit `--lr 1e-5` and silently trained that run at 1e-4. `--lr` is now passed
explicitly on every call and deliberately kept out of `$COMMON`.

**972, not 1000.** 28 conversations lose every supervised token to the 1024-token
truncation and are skipped. `data.py` drops them rather than training on an all-masked
example; `n_train_examples` in each `metrics.json` records the real count.

**One shard, not the whole split.** `data.py` names specific parquet files so `datasets`
doesn't also fetch `train_gen`/`test_gen` (~1.6 GB the experiment never reads). The sample
is therefore drawn from the first `train_sft` shard (~69k conversations), and
`verification_mode="no_checks"` is required because the repo metadata declares four
splits. Widen `SPLIT_FILES` to sample from the full split.

**Batch size 2 × accumulation 8.** Effective batch is still 16. At `--batch-size 4` full
tuning peaks at ~22 GiB and OOMs on a 24 GB card once a batch holds four full-length
sequences. Keep the micro-batch identical across runs or the memory and time columns stop
being comparable.
