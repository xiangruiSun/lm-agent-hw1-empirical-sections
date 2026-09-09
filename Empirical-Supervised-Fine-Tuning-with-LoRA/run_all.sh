#!/usr/bin/env bash
# Runs the full experiment: LoRA r in {1,4,16} plus full fine-tuning, then the
# report and the qualitative generations. Every run shares the same data, seed,
# epochs, batch size and learning rate.
set -euo pipefail
cd "$(dirname "$0")"

# batch-size 2 x grad-accum 8 keeps the effective batch at 16 while leaving
# headroom on a 24 GB card: full fine-tuning peaks at ~22 GiB with batch-size 4,
# which OOMs once a batch happens to hold four full-length sequences.
COMMON="--n-train 1000 --n-val 100 --max-len 1024 --epochs 3 \
        --batch-size 2 --grad-accum 8 --lr 1e-4 --seed 0"

for r in 1 4 16; do
  echo "=== LoRA rank $r ==="
  python train.py --mode lora --rank "$r" --out "runs/lora_r${r}" $COMMON
done

echo "=== Full fine-tuning (lr matched to LoRA) ==="
python train.py --mode full --out runs/full $COMMON

# 1e-4 is high for full-weight tuning. The run above holds the LR fixed for
# comparability, as the question asks; this one shows what full FT does when it
# is not forced onto LoRA's learning rate. Report both.
echo "=== Full fine-tuning (lr 1e-5) ==="
python train.py --mode full --lr 1e-5 --out runs/full_lr1e-5 $COMMON

RUNS="runs/lora_r1 runs/lora_r4 runs/lora_r16 runs/full runs/full_lr1e-5"
python report.py   --runs $RUNS --out results
python generate.py --runs $RUNS --out results/generations.md
