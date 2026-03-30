#!/usr/bin/env python3
"""A small Makemore-style bigram name model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_names(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def split_names(names: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]


def build_vocab(names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set("".join(names)))
    stoi = {".": 0}
    stoi.update({ch: i + 1 for i, ch in enumerate(chars)})
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def name_tokens(name: str) -> list[str]:
    return [".", *name, "."]


def train_bigram(names: list[str], stoi: dict[str, int], smoothing: float) -> np.ndarray:
    counts = np.full((len(stoi), len(stoi)), smoothing, dtype=np.float64)
    for name in names:
        tokens = name_tokens(name)
        for ch1, ch2 in zip(tokens, tokens[1:]):
            counts[stoi[ch1], stoi[ch2]] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def mean_nll(names: list[str], probs: np.ndarray, stoi: dict[str, int]) -> float:
    losses: list[float] = []
    for name in names:
        tokens = name_tokens(name)
        for ch1, ch2 in zip(tokens, tokens[1:]):
            losses.append(-np.log(probs[stoi[ch1], stoi[ch2]]))
    return float(np.mean(losses))


def sample_names(
    probs: np.ndarray,
    itos: dict[int, str],
    rng: np.random.Generator,
    count: int,
) -> list[str]:
    names: list[str] = []
    for _ in range(count):
        idx = 0
        out: list[str] = []
        while True:
            idx = int(rng.choice(probs.shape[1], p=probs[idx]))
            ch = itos[idx]
            if ch == ".":
                break
            out.append(ch)
        names.append("".join(out))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Makemore-style bigram name model.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "names.txt")
    parser.add_argument("--seed", type=int, default=2147483647)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    data_path = args.data.resolve()
    try:
        data_display = data_path.relative_to(Path.cwd().resolve())
    except ValueError:
        data_display = data_path

    names = load_names(data_path)
    train, val, test = split_names(names, args.seed)
    stoi, itos = build_vocab(train)
    probs = train_bigram(train, stoi, args.smoothing)

    val_nll = mean_nll(val, probs, stoi)
    test_nll = mean_nll(test, probs, stoi)
    samples = sample_names(probs, itos, np.random.default_rng(args.seed + 1), args.num_samples)

    print("makemore bigram smoke run")
    print(f"data={data_display}")
    print(f"names={len(names)} train={len(train)} val={len(val)} test={len(test)} vocab={len(stoi)}")
    print(f"smoothing={args.smoothing:g} seed={args.seed}")
    print(f"val_nll={val_nll:.4f} test_nll={test_nll:.4f}")
    print("samples:")
    for name in samples:
        print(f"- {name}")


if __name__ == "__main__":
    main()
