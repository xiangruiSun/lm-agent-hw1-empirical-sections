"""
Collect metrics.json from every run into the table and loss-curve figure for the
write-up.

    python report.py --runs runs/lora_r1 runs/lora_r4 runs/lora_r16 runs/full \
        --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"lora_r1": "#4C72B0", "lora_r4": "#DD8452", "lora_r16": "#55A868",
          "full": "#C44E52", "full_lr1e-5": "#8172B3"}


def label_of(m, show_lr=False):
    base = f"LoRA r={m['rank']}" if m["mode"] == "lora" else "Full fine-tuning"
    return f"{base} (lr={m['config']['lr']:g})" if show_lr else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/lora_r1", "runs/lora_r4",
                                                  "runs/lora_r16", "runs/full"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for r in args.runs:
        p = Path(r) / "metrics.json"
        if not p.exists():
            print(f"skipping missing {p}")
            continue
        runs.append(json.loads(p.read_text()))

    # If the sweep mixes learning rates (e.g. full FT at both 1e-4 and 1e-5),
    # put the LR in every label so no two rows read identically.
    show_lr = len({m["config"]["lr"] for m in runs}) > 1

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for m in runs:
        c = COLORS.get(Path(m["run"]).name, None)
        tr = m["history"]["train"]
        va = m["history"]["val"]
        axes[0].plot([h["step"] for h in tr], [h["loss"] for h in tr],
                     label=label_of(m, show_lr), color=c, lw=1.6)
        axes[1].plot([h["step"] for h in va], [h["loss"] for h in va],
                     label=label_of(m, show_lr), color=c, lw=1.6, marker="o", ms=3)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss (held-out test_sft)")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross-entropy on assistant tokens")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=160)
    print(f"Wrote {out/'loss_curves.png'}")

    # ------------------------------------------------------------------ table
    hdr = ("| Configuration | Trainable params | % of total | Peak GPU mem (GiB) | "
           "Train time (s) | s / step | Final val loss | Val PPL |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    rows = [hdr, sep]
    for m in runs:
        rows.append(
            f"| {label_of(m, show_lr)} | {m['trainable_params']:,} | {m['trainable_pct']:.3f}% | "
            f"{m['peak_gpu_mem_alloc_gib']:.2f} | {m['train_seconds']:.0f} | "
            f"{m['sec_per_step']:.2f} | {m['final_val_loss']:.4f} | {m['final_val_ppl']:.2f} |"
        )
    if runs:
        base = runs[0]["base_val_loss"]
        rows.append(f"| *base model (no fine-tuning)* | 0 | 0% | – | – | – | {base:.4f} | "
                    f"{__import__('math').exp(min(base, 20)):.2f} |")

    table = "\n".join(rows)
    meta = runs[0] if runs else {}
    text = "\n".join([
        "# SFT with LoRA: results",
        "",
        f"Model: `{meta.get('config', {}).get('model', 'Qwen/Qwen2.5-0.5B')}` · "
        f"GPU: {meta.get('gpu', 'n/a')} · "
        f"{meta.get('n_train_examples', '?')} train / {meta.get('n_val_examples', '?')} val "
        f"conversations · {meta.get('config', {}).get('epochs', '?')} epochs · "
        f"effective batch "
        f"{meta.get('config', {}).get('batch_size', 0) * meta.get('config', {}).get('grad_accum', 0)} · "
        f"lr {meta.get('config', {}).get('lr', '?')} · max_len "
        f"{meta.get('config', {}).get('max_len', '?')}",
        "",
        table,
        "",
        "![loss curves](loss_curves.png)",
        "",
    ])
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
