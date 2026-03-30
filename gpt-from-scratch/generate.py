#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHECKPOINT_PATH = Path(__file__).with_name('gpt_model.pt')
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

vocab_size = checkpoint['vocab_size']
stoi      = checkpoint['stoi']
itos      = checkpoint['itos']
block_size = checkpoint['block_size']
n_embd    = checkpoint['n_embd']
n_layer   = checkpoint['n_layer']
n_head    = checkpoint['n_head']
dropout   = checkpoint['dropout']

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v   = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.head_size = n_embd // n_heads
        self.heads   = nn.ModuleList([Head(self.head_size) for _ in range(n_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded from gpt_model.pt  |  device: {device}")


def generate(prompt='', max_new_tokens=500, temperature=1.0, top_k=None):
    if prompt:
        tokens = encode(prompt)
        context = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text with the trained GPT model')
    parser.add_argument('--prompt',     type=str,   default='',    help='Prompt to condition on')
    parser.add_argument('--max_tokens', type=int,   default=500,   help='Number of tokens to generate')
    parser.add_argument('--temperature',type=float, default=1.0,   help='Sampling temperature (lower = more focused)')
    parser.add_argument('--top_k',      type=int,   default=None,  help='Top-k sampling (e.g. 50)')
    args = parser.parse_args()

    text = generate(args.prompt, args.max_tokens, args.temperature, args.top_k)
    print('\n' + '-' * 60)
    print(text)
