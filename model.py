"""
Haroon-2.5-HK
Decoder-only Transformer language model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    vocab_size = 2048
    block_size = 256
    n_embd = 192
    n_head = 6
    n_layer = 4
    dropout = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        b, t, c = x.shape

        q, k, v = self.qkv(x).split(c, dim=2)

        q = q.view(
            b, t, self.n_head, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            b, t, self.n_head, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            b, t, self.n_head, self.head_dim
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (
            self.head_dim ** 0.5
        )

        scores = scores.masked_fill(
            self.mask[:, :, :t, :t] == 0,
            float("-inf"),
        )

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = self.attn_drop(weights)

        y = weights @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)

        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HaroonGPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        self.config = config or GPTConfig()
        c = self.config

        self.token_emb = nn.Embedding(
            c.vocab_size, c.n_embd
        )

        self.pos_emb = nn.Embedding(
            c.block_size, c.n_embd
        )

        self.drop = nn.Dropout(c.dropout)

        self.blocks = nn.Sequential(
            *[Block(c) for _ in range(c.n_layer)]
        )

        self.ln_f = nn.LayerNorm(c.n_embd)

        self.lm_head = nn.Linear(
            c.n_embd, c.vocab_size, bias=False
        )

        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if (
                isinstance(module, nn.Linear)
                and module.bias is not None
            ):
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2:
            raise ValueError(
                "Input must have shape [batch, sequence]"
            )

        if idx.shape[1] == 0:
            raise ValueError("Input sequence cannot be empty")

        if (
            idx.min().item() < 0
            or idx.max().item() >= self.config.vocab_size
        ):
            raise ValueError("Invalid token ID")

        b, t = idx.shape

        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            t = idx.shape[1]

            if targets is not None:
                targets = targets[:, -t:]

        positions = torch.arange(t, device=idx.device)

        x = self.drop(
            self.token_emb(idx)
            + self.pos_emb(positions)[None, :, :]
        )

        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=96,
        temperature=0.8,
        top_k=40,
        repetition_penalty=1.08,
    ):
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError(
                "Input must contain at least one token"
            )

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero"
            )

        self.eval()

        for _ in range(max(0, int(max_new_tokens))):
            context = idx[:, -self.config.block_size:]

            logits, _ = self(context)

            logits = (
                logits[:, -1, :].float() / temperature
            )

            # Reduce the chance of repeating recent bytes.
            if repetition_penalty > 1.0:
                recent = idx[:, -32:]

                for token_id in recent[0].unique():
                    token_id = int(token_id)

                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Top-k sampling.
            if top_k is not None and top_k > 0:
                k = min(int(top_k), logits.size(-1))

                values, _ = torch.topk(logits, k)
                cutoff = values[:, [-1]]

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            probs = F.softmax(logits, dim=-1)

            if (
                not torch.isfinite(probs).all()
                or probs.sum().item() <= 0
            ):
                next_id = logits.argmax(
                    dim=-1,
                    keepdim=True,
                )
            else:
                next_id = torch.multinomial(
                    probs,
                    num_samples=1,
                )

            idx = torch.cat((idx, next_id), dim=1)

        return idx
