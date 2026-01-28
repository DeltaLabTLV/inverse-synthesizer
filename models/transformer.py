# full_transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder


def patchify_spectrogram(spec: torch.Tensor, patch_f: int, patch_t: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    spec: (B, F, T)
    Returns:
      patches: (B, N, patch_dim)
      meta: (F0, T0, n_pf, n_pt)
    Crops to multiples of patch sizes.
    """
    B, F_bins, T_frames = spec.shape
    F0 = (F_bins // patch_f) * patch_f
    T0 = (T_frames // patch_t) * patch_t
    spec = spec[:, :F0, :T0]

    n_pf = F0 // patch_f
    n_pt = T0 // patch_t

    x = spec.view(B, n_pf, patch_f, n_pt, patch_t)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    patches = x.view(B, n_pf * n_pt, patch_f * patch_t)
    return patches, (F0, T0, n_pf, n_pt)


def unpatchify_spectrogram(patches: torch.Tensor, meta: Tuple[int, int, int, int], patch_f: int, patch_t: int) -> torch.Tensor:
    F0, T0, n_pf, n_pt = meta
    B, N, D = patches.shape
    assert N == n_pf * n_pt
    assert D == patch_f * patch_t

    x = patches.view(B, n_pf, n_pt, patch_f, patch_t)
    x = x.permute(0, 1, 3, 2, 4).contiguous()
    spec = x.view(B, F0, T0)
    return spec


@dataclass
class PatchMaskConfig:
    mask_ratio: float = 0.45
    fixed_count: bool = True


def make_patch_mask(
    B: int, N: int, cfg: PatchMaskConfig, device: torch.device, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    if cfg.fixed_count:
        k = int(round(cfg.mask_ratio * N))
        k = max(0, min(N, k))
        mask = torch.zeros((B, N), device=device, dtype=torch.bool)
        for b in range(B):
            perm = torch.randperm(N, generator=generator, device=device)
            mask[b, perm[:k]] = True
        return mask
    return (torch.rand((B, N), device=device, generator=generator) < cfg.mask_ratio)


class SpectrogramMAETransformer(nn.Module):
    """
    Full Transformer MAE-style model:
      - Encoder: contextualizes visible tokens
      - Decoder: reconstructs all patches (loss computed on masked region only)
    """

    def __init__(
        self,
        patch_f: int = 5,
        patch_t: int = 5,
        d_model: int = 512,
        enc_layers: int = 8,
        enc_heads: int = 8,
        enc_ff: int = 2048,
        dec_layers: int = 4,
        dec_heads: int = 8,
        dec_ff: int = 2048,
        dropout: float = 0.1,
        max_tokens: int = 4096,
        decoder_dim: Optional[int] = None,  # allow smaller decoder dim if you want
    ):
        super().__init__()
        self.patch_f = patch_f
        self.patch_t = patch_t
        self.patch_dim = patch_f * patch_t
        self.max_tokens = max_tokens

        self.d_model = d_model
        self.dec_dim = decoder_dim if decoder_dim is not None else d_model

        # Patch embed for encoder
        self.patch_embed = nn.Linear(self.patch_dim, d_model)

        # Pos embeddings
        self.pos_embed_enc = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed_enc, std=0.02)

        # Mask token (encoder space)
        self.mask_token_enc = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token_enc, std=0.02)

        # Encoder
        self.encoder = TransformerEncoder(d_model, enc_layers, enc_heads, enc_ff, dropout=dropout)

        # Map encoder outputs to decoder dim if needed
        self.enc_to_dec = nn.Linear(d_model, self.dec_dim) if self.dec_dim != d_model else nn.Identity()

        # Decoder input token embed (we feed either embedded patches or a decoder mask token)
        self.pos_embed_dec = nn.Parameter(torch.zeros(1, max_tokens, self.dec_dim))
        nn.init.trunc_normal_(self.pos_embed_dec, std=0.02)

        self.mask_token_dec = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        nn.init.trunc_normal_(self.mask_token_dec, std=0.02)

        # If decoder_dim != encoder_dim, project patch embeddings for visible tokens into decoder space
        self.patch_embed_dec = nn.Linear(self.patch_dim, self.dec_dim)

        # Decoder
        self.decoder = TransformerDecoder(self.dec_dim, dec_layers, dec_heads, dec_ff, dropout=dropout)

        # Reconstruction head: decoder tokens -> patch pixels
        self.recon_head = nn.Sequential(
            nn.LayerNorm(self.dec_dim),
            nn.Linear(self.dec_dim, self.patch_dim),
        )

    def forward(
        self,
        spec: torch.Tensor,                    # (B, F, T) normalized
        mask_cfg: Optional[PatchMaskConfig] = None,
        pretrain: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Returns:
          recon_patches: (B, N, patch_dim)
          target_patches: (B, N, patch_dim)
          patch_mask: (B, N) bool
          meta: patch meta
        """
        patches, meta = patchify_spectrogram(spec, self.patch_f, self.patch_t)  # (B,N,patch_dim)
        B, N, _ = patches.shape
        device = patches.device

        if N > self.max_tokens:
            raise ValueError(f"N={N} exceeds max_tokens={self.max_tokens}. Increase max_tokens.")

        # Mask selection (over patches)
        if pretrain and mask_cfg is not None:
            patch_mask = make_patch_mask(B, N, mask_cfg, device=device, generator=generator)  # (B,N)
        else:
            patch_mask = torch.zeros((B, N), device=device, dtype=torch.bool)

        # ----- Encoder tokens -----
        tok_enc = self.patch_embed(patches) + self.pos_embed_enc[:, :N, :]  # (B,N,d_model)
        if pretrain and mask_cfg is not None:
            # replace masked tokens with encoder mask token
            tok_enc = tok_enc.clone()
            tok_enc[patch_mask] = self.mask_token_enc.expand(B, N, -1)[patch_mask]

        z_enc = self.encoder(tok_enc)               # (B,N,d_model)
        memory = self.enc_to_dec(z_enc)             # (B,N,dec_dim)

        # ----- Decoder tokens -----
        # We feed ALL tokens to decoder. Masked ones are replaced by decoder mask token.
        tok_dec = self.patch_embed_dec(patches) + self.pos_embed_dec[:, :N, :]  # (B,N,dec_dim)
        if pretrain and mask_cfg is not None:
            tok_dec = tok_dec.clone()
            tok_dec[patch_mask] = self.mask_token_dec.expand(B, N, -1)[patch_mask]

        z_dec = self.decoder(tok_dec, memory)       # (B,N,dec_dim)

        recon_patches = self.recon_head(z_dec)      # (B,N,patch_dim)
        target_patches = patches                    # (B,N,patch_dim)

        return recon_patches, target_patches, patch_mask, meta

    def reconstruction_loss(
        self,
        recon_patches: torch.Tensor,
        target_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        loss_type: str = "l2",
    ) -> torch.Tensor:
        """
        Compute MAE loss on masked patches only (typical).
        """
        if patch_mask.sum() == 0:
            return recon_patches.sum() * 0.0

        pred = recon_patches[patch_mask]     # (M, patch_dim)
        tgt = target_patches[patch_mask]     # (M, patch_dim)

        if loss_type == "l2":
            return F.mse_loss(pred, tgt)
        if loss_type == "l1":
            return F.l1_loss(pred, tgt)
        raise ValueError(f"Unknown loss_type: {loss_type}")
