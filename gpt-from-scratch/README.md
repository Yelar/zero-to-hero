
# GPT from Scratch

Character-level language modeling on Tiny Shakespeare.

This folder keeps the original lecture-style scripts and adds `scripts/train_char_gpt.py`, a compact CLI trainer for small local runs. It is intentionally sized for smoke tests and educational iteration rather than benchmark chasing.

## Files

- `bigram.py`: first language-model baseline from the lecture path.
- `gpt.py`: original transformer-style script.
- `gpt_dev.ipynb`: notebook exploration.
- `scripts/train_char_gpt.py`: repo-friendly trainer with CLI flags, loss logging, sample generation, and optional checkpoint saving.
- `input.txt`: Tiny Shakespeare corpus.

## Run

```bash
python gpt-from-scratch/scripts/train_char_gpt.py \
  --max-iters 100 \
  --eval-interval 25 \
  --eval-iters 10
```

For a faster smoke test:

```bash
python gpt-from-scratch/scripts/train_char_gpt.py \
  --max-iters 50 \
  --eval-interval 25 \
  --eval-iters 5 \
  --batch-size 16 \
  --block-size 64 \
  --n-embd 64 \
  --n-head 4 \
  --n-layer 2
```

Generated checkpoints go under `gpt-from-scratch/results/generated/` by default and are ignored by Git.
