# Makemore

Character-level name modeling from the Makemore learning path.

## Files

- `names.txt`: training data for name generation.
- `scripts/bigram_names.py`: small reproducible bigram model that runs without PyTorch.
- `makemore_part1_bigrams.ipynb`: notebook exploration for bigram counts and sampling.
- `build_makemore_mlp.ipynb`: MLP character model.
- `makemore_part3_bn.ipynb`: batch norm and training diagnostics.
- `part5.ipynb`: later Makemore work.

## Run

```bash
python makemore/scripts/bigram_names.py --num-samples 20
```

This prints split sizes, negative log-likelihood on validation/test names, and generated sample names.
