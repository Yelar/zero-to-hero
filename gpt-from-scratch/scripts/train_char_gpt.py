#!/usr/bin/env python3
"""Compact character-level GPT trainer for Tiny Shakespeare."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class TrainConfig:
    data: str
    out_dir: str
    seed: int
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    eval_iters: int
    learning_rate: float
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    device: str
    sample_tokens: int
    save_checkpoint: bool


class CharDataset:
    def __init__(self, path: Path, block_size: int, batch_size: int, device: str):
        text = path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda ids: "".join(self.itos[int(i)] for i in ids)
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: TrainConfig):
        super().__init__()
        self.block_size = cfg.block_size
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(
            *[Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(t, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            _, _, c = logits.shape
            loss = F.cross_entropy(logits.reshape(b * t, c), targets.reshape(b * t))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel, data: CharDataset, eval_iters: int) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y = data.get_batch(split)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.item()))
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "input.txt"
    parser = argparse.ArgumentParser(description="Train a tiny character GPT on Tiny Shakespeare.")
    parser.add_argument("--data", default=str(default_data))
    parser.add_argument("--out-dir", default=str(project_root / "results" / "generated"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-tokens", type=int, default=300)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    cfg = TrainConfig(
        data=args.data,
        out_dir=args.out_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        learning_rate=args.learning_rate,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        device=device,
        sample_tokens=args.sample_tokens,
        save_checkpoint=not args.no_save_checkpoint,
    )

    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    data_path = Path(cfg.data).resolve()
    try:
        data_display = data_path.relative_to(Path.cwd().resolve())
    except ValueError:
        data_display = data_path

    torch.manual_seed(cfg.seed)
    data = CharDataset(data_path, cfg.block_size, cfg.batch_size, cfg.device)
    model = GPTLanguageModel(data.vocab_size, cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print("tiny char GPT smoke run")
    print(f"data={data_display}")
    print(f"device={cfg.device} vocab_size={data.vocab_size} params={n_params / 1e6:.3f}M")
    print(
        "config="
        + json.dumps(
            {
                "batch_size": cfg.batch_size,
                "block_size": cfg.block_size,
                "n_embd": cfg.n_embd,
                "n_head": cfg.n_head,
                "n_layer": cfg.n_layer,
                "max_iters": cfg.max_iters,
            },
            sort_keys=True,
        )
    )

    t0 = time.perf_counter()
    for step in range(cfg.max_iters + 1):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters:
            losses = estimate_loss(model, data, cfg.eval_iters)
            elapsed = time.perf_counter() - t0
            print(f"step={step:04d} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} elapsed_s={elapsed:.1f}")
            if step == cfg.max_iters:
                break

        xb, yb = data.get_batch("train")
        _, loss = model(xb, yb)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    sample_ids = model.generate(context, cfg.sample_tokens)[0].tolist()
    sample = data.decode(sample_ids)
    print("sample:")
    print(sample)

    if cfg.save_checkpoint:
        checkpoint_path = out_dir / "tiny_char_gpt.pt"
        torch.save(
            {
                "config": asdict(cfg),
                "model_state_dict": model.state_dict(),
                "stoi": data.stoi,
                "itos": data.itos,
            },
            checkpoint_path,
        )
        try:
            checkpoint_display = checkpoint_path.relative_to(Path.cwd().resolve())
        except ValueError:
            checkpoint_display = checkpoint_path
        print(f"checkpoint={checkpoint_display}")


if __name__ == "__main__":
    main()
