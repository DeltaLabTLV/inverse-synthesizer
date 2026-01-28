import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Patchification (5x5 blocks)
# -----------------------------
def patchify_spectrogram(spec: torch.Tensor, patch_f: int, patch_t: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    spec: (B, F, T)
    Returns:
      patches: (B, N, patch_f*patch_t) flattened
      meta: (F0, T0, n_pf, n_pt) where F0,T0 are cropped sizes used
    Notes:
      - Crops F,T to multiples of patch sizes (simple + deterministic).
      - You can switch to padding if you prefer.
    """
    B, F_bins, T_frames = spec.shape
    F0 = (F_bins // patch_f) * patch_f
    T0 = (T_frames // patch_t) * patch_t
    spec = spec[:, :F0, :T0]

    n_pf = F0 // patch_f
    n_pt = T0 // patch_t
    # (B, n_pf, patch_f, n_pt, patch_t)
    x = spec.view(B, n_pf, patch_f, n_pt, patch_t)
    # (B, n_pf, n_pt, patch_f, patch_t)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    # (B, N, patch_f*patch_t)
    patches = x.view(B, n_pf * n_pt, patch_f * patch_t)
    return patches, (F0, T0, n_pf, n_pt)


def unpatchify_spectrogram(patches: torch.Tensor, meta: Tuple[int, int, int, int], patch_f: int, patch_t: int) -> torch.Tensor:
    """
    patches: (B, N, patch_f*patch_t)
    meta: (F0, T0, n_pf, n_pt)
    Returns:
      spec: (B, F0, T0)
    """
    F0, T0, n_pf, n_pt = meta
    B, N, D = patches.shape
    assert N == n_pf * n_pt
    assert D == patch_f * patch_t

    x = patches.view(B, n_pf, n_pt, patch_f, patch_t)
    x = x.permute(0, 1, 3, 2, 4).contiguous()  # (B, n_pf, patch_f, n_pt, patch_t)
    spec = x.view(B, F0, T0)
    return spec


# -----------------------------
# MAE-style masking over patches
# -----------------------------
@dataclass
class PatchMaskConfig:
    mask_ratio: float = 0.45
    # If True, ensures exactly round(mask_ratio * N) masked tokens per sample
    fixed_count: bool = True


def make_patch_mask(
    B: int,
    N: int,
    cfg: "PatchMaskConfig",
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Returns:
      mask: (B, N) bool where True=masked
    """
    if cfg.fixed_count:
        k = int(round(cfg.mask_ratio * N))
        k = max(0, min(N, k))
        mask = torch.zeros((B, N), device=device, dtype=torch.bool)
        for b in range(B):
            perm = torch.randperm(N, generator=generator, device=device)
            mask[b, perm[:k]] = True
        return mask
    else:
        return (torch.rand((B, N), device=device, generator=generator) < cfg.mask_ratio)


# -----------------------------
# Transformer blocks (Vaswani-style)
# -----------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (pre-norm)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop1(h)

        # FFN (pre-norm)
        h = self.ln2(x)
        h = self.ff(h)
        x = x + self.drop2(h)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.ln_out(x)


# -----------------------------
# Full model: Patch embed + pos embed + mask token + projection heads
# -----------------------------
class SpectrogramTransformerContrastive(nn.Module):
    """
    Pretraining forward returns:
      z_ctx: contextual embeddings (B, N, d_model)
      z_tgt: target embeddings (B, N, d_model) from clean patches (stop-grad option outside)
      patch_mask: (B, N) bool
      meta: patch meta for (F0,T0,n_pf,n_pt)
    """

    def __init__(
        self,
        patch_f: int = 5,
        patch_t: int = 5,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_tokens: int = 4096,
        proj_dim: Optional[int] = None,   # if set, use projection head to proj_dim for contrastive
    ):
        super().__init__()
        self.patch_f = patch_f
        self.patch_t = patch_t
        patch_dim = patch_f * patch_t

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = TransformerEncoder(d_model, n_layers, n_heads, d_ff, dropout=dropout)

        # Projection heads (optional, common in contrastive learning)
        out_dim = proj_dim if proj_dim is not None else d_model
        self.proj_q = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )
        self.proj_k = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(
        self,
        spec: torch.Tensor,                  # (B, F, T) normalized
        mask_cfg: Optional[PatchMaskConfig] = None,
        pretrain: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        patches, meta = patchify_spectrogram(spec, self.patch_f, self.patch_t)  # (B,N,patch_dim)
        B, N, _ = patches.shape
        device = patches.device

        if N > self.pos_embed.shape[1]:
            raise ValueError(f"N={N} tokens exceeds max_tokens={self.pos_embed.shape[1]} (increase max_tokens).")

        # Token embeddings from clean patches (targets)
        tok_clean = self.patch_embed(patches)                               # (B,N,d_model)
        tok_clean = tok_clean + self.pos_embed[:, :N, :]                    # add pos

        if pretrain and mask_cfg is not None:
            patch_mask = make_patch_mask(B, N, mask_cfg, device=device, generator=generator)  # (B,N)
            # Replace masked tokens with learnable mask token (plus pos)
            tok_in = tok_clean.clone()
            tok_in[patch_mask] = self.mask_token.expand(B, N, -1)[patch_mask]
        else:
            patch_mask = torch.zeros((B, N), device=device, dtype=torch.bool)
            tok_in = tok_clean

        # Contextual encoding
        z_ctx = self.encoder(tok_in)                                        # (B,N,d_model)

        # Prepare contrastive spaces
        q = self.proj_q(z_ctx)                                               # (B,N,D)
        k = self.proj_k(tok_clean)                                           # (B,N,D)  (targets from clean tokens)

        return q, k, patch_mask, meta


def masked_infonce_loss(
    q: torch.Tensor,              # (B, N, D) queries from contextual encoder
    k: torch.Tensor,              # (B, N, D) targets (clean)
    mask: torch.Tensor,           # (B, N) bool True=masked positions to score
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE over masked positions:
      loss = -log exp(sim(q_i, k_i)/tau) / sum_j exp(sim(q_i, k_j)/tau)
    where j ranges over all tokens in batch (including itself; that's fine because it's the positive).
    """
    B, N, D = q.shape
    assert k.shape == (B, N, D)
    assert mask.shape == (B, N)

    # Select masked queries
    q_m = q[mask]   # (M, D)
    k_pos = k[mask] # (M, D)
    if q_m.numel() == 0:
        # No masked tokens => loss 0 (avoid NaNs)
        return q.sum() * 0.0

    # Normalize for cosine similarity
    q_m = F.normalize(q_m, dim=-1)
    k_all = F.normalize(k.view(B * N, D), dim=-1)
    k_pos = F.normalize(k_pos, dim=-1)

    # logits: (M, B*N)
    logits = (q_m @ k_all.t()) / temperature

    # labels: index of the positive key in flattened k_all
    # We need the flattened indices of masked positions.
    flat_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)  # (M,)
    labels = flat_idx

    loss = F.cross_entropy(logits, labels)
    return loss
