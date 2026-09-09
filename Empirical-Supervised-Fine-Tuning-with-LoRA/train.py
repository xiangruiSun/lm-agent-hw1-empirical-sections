"""
Supervised fine-tuning of Qwen/Qwen2.5-0.5B on UltraChat, with LoRA or full weights.

One run = one configuration. Everything except the parameterization (LoRA rank, or
full fine-tuning) is held fixed: same 1000 training conversations, same held-out
set, same epochs, batch size, learning rate, schedule, sequence length and seed.

Examples
--------
    python train.py --mode lora --rank 1  --out runs/lora_r1
    python train.py --mode lora --rank 4  --out runs/lora_r4
    python train.py --mode lora --rank 16 --out runs/lora_r16
    python train.py --mode full           --out runs/full

Writes to <out>/:
    metrics.json   trainable params, peak GPU memory, wall-clock time, loss curves
    adapter/       LoRA adapter (mode=lora) or full model weights (mode=full)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from data import Collator, SFTDataset

IGNORE_INDEX = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--mode", choices=["lora", "full"], required=True)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank (mode=lora)")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="Default 2*rank, which holds the LoRA scaling alpha/r fixed "
                        "across ranks so the comparison isolates rank.")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--out", required=True)

    # Held fixed across all runs.
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--streaming", action="store_true",
                   help="Stream the dataset instead of caching it to disk. Same "
                        "seed-determined subset; use when disk is tight.")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4, help="per-device micro-batch")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5, help="optimizer steps")
    p.add_argument("--eval-every", type=int, default=20, help="optimizer steps")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1, help="debug: cap optimizer steps")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def batch_loss(model, batch):
    """Returns (mean loss over supervised tokens, number of supervised tokens)."""
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    n_sup = (batch["labels"][:, 1:] != IGNORE_INDEX).sum()
    return out.loss, n_sup


@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx):
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_sum = torch.zeros((), device=device, dtype=torch.float64)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx():
            loss, n_sup = batch_loss(model, batch)
        # HF returns the mean over supervised tokens; re-weight to get a proper
        # token-level average across the whole validation set.
        loss_sum += loss.double() * n_sup
        tok_sum += n_sup
    model.train()
    return (loss_sum / tok_sum).item()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    use_amp = device.type == "cuda" and args.dtype != "fp32"

    def autocast_ctx():
        if use_amp:
            return torch.autocast("cuda", dtype=torch_dtype)
        return torch.autocast("cpu", enabled=False)

    # ---------------------------------------------------------------- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------------- data
    print("Building datasets ...", flush=True)
    train_ds = SFTDataset(tokenizer, "train_sft", args.n_train, args.max_len,
                          seed=args.seed, streaming=args.streaming)
    val_ds = SFTDataset(tokenizer, "test_sft", args.n_val, args.max_len,
                        seed=args.seed, streaming=args.streaming)
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"train_tokens={train_ds.token_stats()}", flush=True)

    collate = Collator(tokenizer.pad_token_id)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=g, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    # ---------------------------------------------------------------- model
    # Weights are kept in fp32 and compute is done under bf16 autocast, so the
    # memory numbers reflect a standard mixed-precision setup in both modes.
    try:  # `dtype` in transformers>=5, `torch_dtype` before that
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.use_cache = False

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.rank
        peft_config = LoraConfig(
            r=args.rank,
            lora_alpha=alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",  # every linear layer in the transformer
        )
        model = get_peft_model(model, peft_config)
        adapted = sorted({n.split(".lora_A")[0].split(".")[-1]
                          for n, _ in model.named_parameters() if "lora_A" in n})
        print(f"LoRA r={args.rank} alpha={alpha} adapted modules: {adapted}", flush=True)

    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    trainable, total = count_params(model)
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)", flush=True)

    # ---------------------------------------------------------------- optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and args.dtype == "fp16"))
    print(f"optimizer steps: {total_steps} ({steps_per_epoch}/epoch, "
          f"effective batch {args.batch_size * args.grad_accum})", flush=True)

    # ---------------------------------------------------------------- train
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    history = {"train": [], "val": []}
    step = 0
    running_loss, running_tok = 0.0, 0
    model.train()

    # Step 0 baseline: the loss of the un-finetuned model on both splits.
    history["val"].append({"step": 0, "loss": evaluate(model, val_loader, device, autocast_ctx)})
    print(f"[step 0] val_loss={history['val'][0]['loss']:.4f}", flush=True)

    t0 = time.perf_counter()
    done = False
    epoch = 0
    while not done:
        epoch += 1
        optimizer.zero_grad(set_to_none=True)  # don't carry a partial group across epochs
        for micro, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx():
                loss, n_sup = batch_loss(model, batch)
            scaler.scale(loss / args.grad_accum).backward()

            running_loss += loss.item() * n_sup.item()
            running_tok += n_sup.item()

            if (micro + 1) % args.grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                tr = running_loss / max(running_tok, 1)
                history["train"].append({"step": step, "epoch": step / steps_per_epoch,
                                         "loss": tr, "lr": scheduler.get_last_lr()[0]})
                running_loss, running_tok = 0.0, 0
                print(f"[step {step}/{total_steps}] train_loss={tr:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

            if step % args.eval_every == 0 or step == total_steps:
                vl = evaluate(model, val_loader, device, autocast_ctx)
                history["val"].append({"step": step, "epoch": step / steps_per_epoch, "loss": vl})
                print(f"[step {step}/{total_steps}] val_loss={vl:.4f}", flush=True)

            if step >= total_steps:
                done = True
                break

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30 if device.type == "cuda" else 0.0

    # ---------------------------------------------------------------- save
    model.config.use_cache = True
    # PEFT writes only the adapter weights; full fine-tuning writes the whole model.
    save_dir = out_dir / ("adapter" if args.mode == "lora" else "model")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    final_val = history["val"][-1]["loss"]
    metrics = {
        "run": out_dir.name,
        "mode": args.mode,
        "save_dir": str(save_dir),
        "rank": args.rank if args.mode == "lora" else None,
        "lora_alpha": (args.lora_alpha if args.lora_alpha is not None else 2 * args.rank)
                      if args.mode == "lora" else None,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "peak_gpu_mem_alloc_gib": peak_alloc,
        "peak_gpu_mem_reserved_gib": peak_reserved,
        "train_seconds": train_seconds,
        "sec_per_step": train_seconds / max(step, 1),
        "optimizer_steps": step,
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(min(final_val, 20)),
        "best_val_loss": min(h["loss"] for h in history["val"]),
        "base_val_loss": history["val"][0]["loss"],
        "history": history,
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_ds),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
