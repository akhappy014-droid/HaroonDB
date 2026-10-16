"""
============================================================
Haroon-2.5-HK
Larger Decoder-Only Transformer Language Model

Architecture:
    Layers:       6
    Attention:    8 heads
    Embedding:    256
    Context:      256 tokens
    Vocabulary:   256 UTF-8 bytes

IMPORTANT:
    This architecture requires training from scratch.
    Old checkpoints are not compatible.
============================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Configuration
# ============================================================

class GPTConfig:
    vocab_size = 256
    block_size = 256

    n_embd = 256
    n_head = 8
    n_layer = 6

    dropout = 0.1


# ============================================================
# Causal Self Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
        )

        self.proj = nn.Linear(
            config.n_embd,
            config.n_embd,
        )

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                    dtype=torch.bool,
                )
            ).view(
                1,
                1,
                config.block_size,
                config.block_size,
            ),
            persistent=False,
        )

    def forward(self, x):

        batch_size, seq_len, channels = x.shape

        q, k, v = self.qkv(x).split(
            channels,
            dim=2,
        )

        q = q.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_head,
            self.head_dim,
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        scores = scores.masked_fill(
            ~self.mask[:, :, :seq_len, :seq_len],
            float("-inf"),
        )

        weights = F.softmax(
            scores.float(),
            dim=-1,
        ).to(q.dtype)

        weights = self.attn_drop(weights)

        output = weights @ v

        output = output.transpose(
            1,
            2,
        ).contiguous().view(
            batch_size,
            seq_len,
            channels,
        )

        output = self.proj(output)

        return self.resid_drop(output)


# ============================================================
# Feed Forward Network
# ============================================================

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                config.n_embd,
                4 * config.n_embd,
            ),

            nn.GELU(),

            nn.Linear(
                4 * config.n_embd,
                config.n_embd,
            ),

            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Transformer Block
# ============================================================

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


# ============================================================
# Haroon GPT
# ============================================================

class HaroonGPT(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        self.config = config or GPTConfig()

        config = self.config

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.pos_emb = nn.Embedding(
            config.block_size,
            config.n_embd,
        )

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(
            *[
                Block(config)
                for _ in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        # Tie token embeddings and output weights.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    # --------------------------------------------------------
    # Weight initialization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, idx, targets=None):

        if idx.ndim != 2:
            raise ValueError(
                "Input must have shape [batch, sequence]"
            )

        if idx.shape[1] == 0:
            raise ValueError(
                "Input sequence cannot be empty"
            )

        if (
            idx.min().item() < 0
            or idx.max().item() >= self.config.vocab_size
        ):
            raise ValueError(
                "Input contains invalid token IDs"
            )

        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:

            idx = idx[:, -self.config.block_size:]

            seq_len = idx.shape[1]

            if targets is not None:
                targets = targets[:, -seq_len:]

        positions = torch.arange(
            seq_len,
            device=idx.device,
        )

        token_embeddings = self.token_emb(idx)

        position_embeddings = self.pos_emb(positions)

        x = self.drop(
            token_embeddings
            + position_embeddings.unsqueeze(0)
        )

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            if targets.shape != idx.shape:
                raise ValueError(
                    "Targets must match the input shape"
                )

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),

                targets.reshape(-1),
            )

        return logits, loss

    # --------------------------------------------------------
    # Text generation
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=96,
        temperature=0.8,
        top_k=40,
    ):

        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError(
                "Input must contain at least one token"
            )

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero"
            )

        if top_k is not None and top_k < 1:
            raise ValueError(
                "top_k must be positive or None"
            )

        was_training = self.training

        self.eval()

        try:

            for _ in range(max(0, int(max_new_tokens))):

                context = idx[:, -self.config.block_size:]

                logits, _ = self(context)

                logits = logits[:, -1, :].float()

                logits = logits / temperature

                # Keep only the most likely tokens.
                if top_k is not None:

                    k = min(
                        int(top_k),
                        logits.size(-1),
                    )

                    values, _ = torch.topk(
                        logits,
                        k,
                        dim=-1,
                    )

                    cutoff = values[:, [-1]]

                    logits = logits.masked_fill(
                        logits < cutoff,
                        float("-inf"),
                    )

                probabilities = F.softmax(
                    logits,
                    dim=-1,
                )

                # Defensive fallback if probabilities are invalid.
                if not torch.isfinite(probabilities).all():

                    next_token = logits.argmax(
                        dim=-1,
                        keepdim=True,
                    )

                else:

                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                    )

                idx = torch.cat(
                    (idx, next_token),
                    dim=1,
                )

            return idx

        finally:
            self.train(was_training)
