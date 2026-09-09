"""
Qualitative comparison: base Qwen2.5-0.5B vs. each fine-tuned model.

Uses held-out first-turn user prompts from ultrachat_200k/test_sft (never seen in
training) plus a few short fixed instructions. Every model gets the identical
prompt, chat template and decoding settings, so any difference in the output is
attributable to fine-tuning.

    python generate.py --runs runs/lora_r1 runs/lora_r4 runs/lora_r16 runs/full \
        --out results/generations.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_split

EXTRA_PROMPTS = [
    "Explain what a hash table is to someone who has never programmed.",
    "Write a short, polite email asking my landlord to fix a leaking tap.",
    "What are three things I should consider before adopting a cat?",
]


def load_prompts(n_from_dataset: int, seed: int, streaming: bool = False):
    prompts = []
    if n_from_dataset > 0:
        # Same seed and split as validation, so these prompts are held out.
        ds = load_split("test_sft", n_from_dataset, seed, streaming)
        for ex in ds:
            first_user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
            prompts.append(first_user.strip())
    return prompts + EXTRA_PROMPTS


def load_model(base_model: str, run_dir: str | None, device):
    if run_dir is None:
        model = AutoModelForCausalLM.from_pretrained(base_model)
        return model.to(device).eval()

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    save_dir = metrics.get("save_dir") or str(Path(run_dir) / "adapter")
    if metrics["mode"] == "lora":
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = PeftModel.from_pretrained(model, save_dir)
        model = model.merge_and_unload()  # fold the adapter in for fast generation
    else:
        model = AutoModelForCausalLM.from_pretrained(save_dir)
    return model.to(device).eval()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device, max_new_tokens: int, greedy: bool, seed: int):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=None if greedy else 0.7,
        top_p=None if greedy else 0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--runs", nargs="+", default=["runs/lora_r1", "runs/lora_r4",
                                                  "runs/lora_r16", "runs/full"])
    ap.add_argument("--out", default="results/generations.md")
    ap.add_argument("--n-prompts", type=int, default=3, help="held-out prompts from test_sft")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = load_prompts(args.n_prompts, args.seed, args.streaming)

    # model label -> {prompt: completion}
    results: dict[str, list[str]] = {}
    for label, run_dir in [("base (no fine-tuning)", None)] + [(Path(r).name, r) for r in args.runs]:
        print(f"--- generating with {label} ---", flush=True)
        model = load_model(args.base_model, run_dir, device)
        results[label] = [
            generate(model, tokenizer, p, device, args.max_new_tokens, not args.sample, args.seed)
            for p in prompts
        ]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qualitative comparison on held-out prompts",
        "",
        f"Decoding: {'sampling (T=0.7, top-p=0.9)' if args.sample else 'greedy'}, "
        f"max_new_tokens={args.max_new_tokens}. Identical prompt and chat template for every model.",
        "",
    ]
    for i, prompt in enumerate(prompts):
        lines += [f"## Prompt {i+1}", "", "> " + prompt.replace("\n", "\n> "), ""]
        for label, completions in results.items():
            lines += [f"### {label}", "", "```", completions[i], "```", ""]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
