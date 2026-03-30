#!/usr/bin/env python
# coding: utf-8

# # GPT from Scratch
# ### Following Karpathy's "Let's build GPT" — commit by commit

# In[294]:


import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path


# In[295]:


# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_layer = 6
dropout = 0.2
n_head = 6
torch.manual_seed(1337)
print("Device:", device)

# ## Data loading (boilerplate — nothing to implement here)

# In[296]:


DATA_PATH = Path(__file__).with_name('input.txt')
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"vocab_size: {vocab_size}")


# In[297]:


class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


# In[298]:


class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)  # You can adjust the dropout rate as needed

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1] **-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)      # add dropout after softmax

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# In[299]:


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads):
        super().__init__()
        self.head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# In[300]:


class Block(nn.Module):

    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.ffwd = FeedForward(n_embd)
        self.multi_head_attention = MultiHeadAttention(n_heads)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x));
        return x


# In[301]:


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.head = Head(n_embd)
        # self.multi_head_attention = MultiHeadAttention(4)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)   # (B, T, n_embd)

        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits  = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            # logits: (B, T, vocab_size), targets: (B, T)
           
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# In[302]:


# Training loop (run this after implementing the model above)
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
    'block_size': block_size,
    'n_embd': n_embd,
    'n_layer': n_layer,
    'n_head': n_head,
    'dropout': dropout,
}, 'gpt_model.pt')
print("Model saved to gpt_model.pt")
