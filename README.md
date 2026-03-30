# Zero to Hero

Follow-along code and experiments from Andrej Karpathy's Neural Networks: Zero to Hero series, plus a small GPT trainer.

- `micrograd/`: scalar autograd and tiny neural nets.
- `makemore/`: character-level name modeling notebooks and a runnable bigram script.
- `gpt-from-scratch/`: Tiny Shakespeare GPT scripts and a compact trainer.

Each folder keeps its own scripts and results.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
python micrograd/scripts/check_micrograd.py
python makemore/scripts/bigram_names.py --num-samples 20
python gpt-from-scratch/scripts/train_char_gpt.py --max-iters 50 --eval-interval 25 --eval-iters 5
```

## Run Residues

Small text outputs are kept inside each project folder:

- `micrograd/results/micrograd_gradient_check_2026-06-08.txt`
- `makemore/results/makemore_bigram_smoke_2026-06-08.txt`
- `gpt-from-scratch/results/gpt_tiny_shakespeare_smoke_2026-06-08.txt`

Large local artifacts are intentionally ignored: checkpoints, virtual environments, and generated model files.
