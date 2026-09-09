"""
Data pipeline for SFT on HuggingFaceH4/ultrachat_200k.

Key detail: we train only on the *assistant* tokens. User turns, system turns and
all chat-template control tokens are masked out with -100 so the loss is a pure
"given the conversation so far, produce this assistant reply" objective.

The masking is done by exploiting the fact that chat templates are prefix
consistent: tokenizing messages[:i] with add_generation_prompt=True gives exactly
the prefix of tokenizing messages[:i+1], so the difference is the assistant span.
"""

from __future__ import annotations

import torch
from datasets import load_dataset

IGNORE_INDEX = -100


def _apply_template(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    """Tokenize a message list with the chat template, returning a flat list of ids.

    Normalizes across transformers versions, which variously return a list, a list
    of lists, or a BatchEncoding.
    """
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    # transformers>=5 returns a BatchEncoding (a UserDict, not a dict subclass);
    # older versions return a plain list of ids.
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return list(out)


def encode_conversation(tokenizer, messages, max_len: int):
    """Return (input_ids, labels) with loss only on assistant tokens, or None."""
    # Drop trailing turns that are not assistant replies: nothing to supervise there.
    while messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    if not messages:
        return None

    input_ids: list[int] = []
    labels: list[int] = []

    for i, msg in enumerate(messages):
        if msg["role"] != "assistant" or i == 0:
            continue
        prompt_ids = _apply_template(tokenizer, messages[:i], add_generation_prompt=True)
        full_ids = _apply_template(tokenizer, messages[: i + 1], add_generation_prompt=False)

        if full_ids[: len(prompt_ids)] != prompt_ids:
            # Template is not prefix-consistent for this tokenizer; bail out loudly
            # rather than silently training on a misaligned mask.
            raise RuntimeError(
                "Chat template is not prefix-consistent; assistant-span masking "
                "cannot be derived this way for this tokenizer."
            )

        # Everything between the end of the previous span and the start of this
        # assistant reply is context -> masked.
        labels.extend([IGNORE_INDEX] * (len(prompt_ids) - len(labels)))
        labels.extend(full_ids[len(prompt_ids):])
        input_ids = full_ids

    if not input_ids:
        return None
    assert len(input_ids) == len(labels)

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if all(l == IGNORE_INDEX for l in labels):
        return None  # truncation removed every supervised token
    return input_ids, labels


REPO = "HuggingFaceH4/ultrachat_200k"

# Only the SFT splits are wanted. Naming the files explicitly stops `datasets`
# from also fetching train_gen/test_gen (~1.6 GB of parquet, plus the Arrow
# conversion of all four splits) that this experiment never touches. One shard
# of train_sft is ~69k conversations -- far more than the 1000 we sample.
SPLIT_FILES = {
    "train_sft": "data/train_sft-00000-of-00003-*.parquet",
    "test_sft": "data/test_sft-*.parquet",
}


def load_split(split: str, n_examples: int, seed: int, streaming: bool):
    """Fixed, seed-determined subset of a split. Identical for every run."""
    # verification_mode="no_checks": the repo's metadata declares four splits and
    # the recorded example counts for whole splits, so `datasets` would reject
    # this deliberately partial download. Skipping that check is the intent.
    kwargs = dict(
        data_files={split: SPLIT_FILES[split]},
        split=split,
        verification_mode="no_checks",
    )
    if streaming:
        # Nothing is written to the datasets cache; only the bytes actually read
        # come down the wire. Use this when disk is tight.
        import itertools

        ds = load_dataset(REPO, streaming=True, **kwargs)
        ds = ds.shuffle(seed=seed, buffer_size=max(2_000, 4 * n_examples))
        return list(itertools.islice(ds, n_examples))

    ds = load_dataset(REPO, **kwargs)
    return ds.shuffle(seed=seed).select(range(min(n_examples, len(ds))))


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, split: str, n_examples: int, max_len: int,
                 seed: int = 0, streaming: bool = False):
        raw = load_split(split, n_examples, seed, streaming)

        self.examples = []
        n_skipped = 0
        for ex in raw:
            enc = encode_conversation(tokenizer, list(ex["messages"]), max_len)
            if enc is None:
                n_skipped += 1
                continue
            self.examples.append(enc)
        self.n_skipped = n_skipped

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels = self.examples[idx]
        return {"input_ids": input_ids, "labels": labels}

    def token_stats(self):
        total = sum(len(x[0]) for x in self.examples)
        supervised = sum(sum(1 for l in x[1] if l != IGNORE_INDEX) for x in self.examples)
        return {"total_tokens": total, "supervised_tokens": supervised}


class Collator:
    """Right-pads a batch; pads labels with -100 so padding never contributes loss."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)
            attn.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
