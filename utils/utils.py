# load_pretrain_weights.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import torch


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Tries common checkpoint formats.
    """
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    # assume it's already a state_dict
    return ckpt


def load_pretrained(
    model: torch.nn.Module,
    ckpt_path: str,
    device: str = "cpu",
    strict: bool = False,
    prefix_in_ckpt: Optional[str] = None,
    prefix_in_model: Optional[str] = None,
) -> Tuple[list, list]:
    """
    Generic load with optional prefix mapping.

    Examples:
      - encoder-only ckpt saved with keys like "encoder.blocks.0...."
        prefix_in_ckpt=None, prefix_in_model=None
      - ckpt keys like "module.encoder...."  -> prefix_in_ckpt="module."
      - want to load into submodule, e.g. model.encoder.*:
        prefix_in_model="encoder."
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = _extract_state_dict(ckpt)

    new_sd = {}
    for k, v in sd.items():
        kk = k
        if prefix_in_ckpt and kk.startswith(prefix_in_ckpt):
            kk = kk[len(prefix_in_ckpt):]
        if prefix_in_model:
            kk = prefix_in_model + kk
        new_sd[kk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    return missing, unexpected


def load_encoder_pretrain_into_full_mae(
    full_model: torch.nn.Module,
    ckpt_path: str,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[list, list]:
    """
    Convenience loader when pretrain checkpoint is encoder-style and full model has:
      - patch_embed
      - pos_embed_enc
      - mask_token_enc
      - encoder
    and you want to ignore decoder weights if missing.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = _extract_state_dict(ckpt)

    # Direct attempt first (if checkpoint already uses full_model keys)
    missing, unexpected = full_model.load_state_dict(sd, strict=False)

    # If too many unexpected keys, try to map "encoder."-relative ckpt into "encoder." of full model
    # Common case: ckpt saved from encoder-only model with keys like "patch_embed.*", "pos_embed.*", "encoder.*"
    # Our full model expects "patch_embed.*", "pos_embed_enc.*", "mask_token_enc.*", "encoder.*"
    # We'll do light key mapping:
    mapped = {}
    for k, v in sd.items():
        kk = k
        kk = kk.replace("pos_embed", "pos_embed_enc")
        kk = kk.replace("mask_token", "mask_token_enc")
        mapped[kk] = v

    missing2, unexpected2 = full_model.load_state_dict(mapped, strict=strict)
    # Return the second result (more relevant)
    return missing2, unexpected2



