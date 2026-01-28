# transformer_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    """
    Vaswani-style Transformer decoder block:
      - self-attention on decoder tokens
      - cross-attention to encoder memory
      - FFN
    All with residual + LayerNorm (pre-norm).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop2 = nn.Dropout(dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,               # (B, N_dec, D)
        memory: torch.Tensor,          # (B, N_enc, D)
        self_attn_mask: Optional[torch.Tensor] = None,
        memory_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,  # optional padding mask (B, N)
    ) -> torch.Tensor:
        # self-attn
        h = self.ln1(x)
        h, _ = self.self_attn(
            h, h, h,
            attn_mask=self_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(h)

        # cross-attn (queries from decoder, keys/values from encoder memory)
        h = self.ln2(x)
        h, _ = self.cross_attn(
            h, memory, memory,
            attn_mask=memory_attn_mask,
            key_padding_mask=None,  # if you have encoder padding mask, pass it here
            need_weights=False,
        )
        x = x + self.drop2(h)

        # ffn
        h = self.ln3(x)
        h = self.ff(h)
        x = x + self.drop3(h)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,               # (B, N_dec, D)
        memory: torch.Tensor,          # (B, N_enc, D)
        self_attn_mask: Optional[torch.Tensor] = None,
        memory_attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(
                x=x,
                memory=memory,
                self_attn_mask=self_attn_mask,
                memory_attn_mask=memory_attn_mask,
                key_padding_mask=key_padding_mask,
            )
        return self.ln_out(x)
