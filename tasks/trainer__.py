import os
import math
import time
import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# ---------- your modules ----------
# Data
# from dataset.data_stft import (
# from dataset.dataloader import (
#     make_files,
#     AudioToStftDataset,
#     StftConfig,
#     MaskConfig,
#     compute_dataset_mean_std,
#     collate_pad_time,
# )
from dataset.dataloader import (
    StftConfig,
    BinMaskConfig,
    make_files,
    make_loader,
    compute_dataset_mean_std,
)


# Models
from models.unet_1d import UNet1DTime
from models.transformer_encoder import SpectrogramTransformerContrastive, PatchMaskConfig as TokenMaskCfg, masked_infonce_loss
from models.transformer import SpectrogramMAETransformer, PatchMaskConfig as MaePatchMaskCfg

# Loss pack (spectral + lowmag + params if needed)
from losses.losses import InverSynth2Loss, SpectralLossWeights, LowMagConfig, ParamLossWeights


# -----------------------------
# Small utilities
# -----------------------------
def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_checkpoint(path: str, model: nn.Module, optim: torch.optim.Optimizer,
                    step: int, extra: Optional[Dict[str, Any]] = None):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None,
                    strict: bool = False, map_location: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if len(unexpected) > 0:
        print("[load_checkpoint] unexpected keys (first 20):", unexpected[:20])
    if len(missing) > 0:
        print("[load_checkpoint] missing keys (first 20):", missing[:20])
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("step", 0))


def build_optimizer(name: str, params, lr: float, weight_decay: float, betas=(0.9, 0.999), eps=1e-8):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    raise ValueError(f"Unknown optimizer: {name}")


def cosine_schedule(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float):
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def apply_lr(optim: torch.optim.Optimizer, lr: float):
    for pg in optim.param_groups:
        pg["lr"] = lr


# -----------------------------
# Config (simple dict-based)
# -----------------------------
def load_config_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Data builder
# -----------------------------
def build_loader(cfg: Dict[str, Any], pretrain: bool, use_masking: bool, mean_std: Tuple[float, float]):
    root = cfg["paths"]["data_root"]
    files = make_files(root)

    stft_cfg = StftConfig(**cfg["stft"])
    mcfg = cfg["masking"].copy()
    mcfg["enable"] = bool(use_masking)
    mask_cfg = MaskConfig(**mcfg)

    sr = cfg["audio"]["target_sr"]
    clip_seconds = cfg["audio"]["clip_seconds"]
    clip_samples = int(round(sr * clip_seconds)) if (sr is not None and clip_seconds is not None) else None

    ds = AudioToStftDataset(
        files=files,
        stft_cfg=stft_cfg,
        mask_cfg=mask_cfg,
        pretrain=pretrain,
        target_sr=sr,
        clip_samples=clip_samples,
        mean_std=mean_std,
        seed=cfg.get("seed", 1234),
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=cfg["data"]["shuffle"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        drop_last=cfg["data"]["drop_last"],
        collate_fn=collate_pad_time,
    )


def compute_or_get_stats(cfg: Dict[str, Any]) -> Tuple[float, float]:
    norm = cfg["normalization"]
    if norm.get("use_fixed_stats", False):
        return float(norm["mean"]), float(norm["std"])

    if not norm.get("compute_if_missing", True):
        # fallback
        return 0.0, 1.0

    root = cfg["paths"]["data_root"]
    files = make_files(root)
    stft_cfg = StftConfig(**cfg["stft"])
    sr = cfg["audio"]["target_sr"]
    clip_seconds = cfg["audio"]["clip_seconds"]
    clip_samples = int(round(sr * clip_seconds)) if (sr is not None and clip_seconds is not None) else None
    max_files = norm.get("max_files_for_stats", None)

    mean, std = compute_dataset_mean_std(
        files=files,
        stft_cfg=stft_cfg,
        target_sr=sr,
        clip_samples=clip_samples,
        max_files=max_files,
    )
    return float(mean), float(std)


# -----------------------------
# Train steps per experiment
# -----------------------------
@torch.no_grad()
def _maybe_print_shapes(batch: Dict[str, Any], step: int):
    if step == 0:
        x, y, m = batch["x"], batch["y"], batch["mask"]
        print(f"[batch] x={tuple(x.shape)} y={tuple(y.shape)} mask={tuple(m.shape)}")


def train_pretrain_autoencoder(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp_enabled = bool(cfg["train"].get("amp", False))

    mean_std = compute_or_get_stats(cfg)
    loader = build_loader(cfg, pretrain=True, use_masking=True, mean_std=mean_std)

    # Model
    F_bins = cfg["models"]["autoencoder_unet"]["in_channels"]
    model = UNet1DTime(**cfg["models"]["autoencoder_unet"]).to(device)

    # Loss (spectral only is fine; you can enable lowmag too)
    loss_cfg = cfg["loss"]
    loss_fn = InverSynth2Loss(
        spec_w=SpectralLossWeights(**loss_cfg["spec"]),
        lowmag=LowMagConfig(**loss_cfg["lowmag"]),
        param_w=ParamLossWeights(**loss_cfg["params"]),
    ).to(device)

    # Optim
    optim_cfg = cfg["optim"]
    optim = build_optimizer(optim_cfg["name"], model.parameters(), optim_cfg["lr"],
                            optim_cfg["weight_decay"], tuple(optim_cfg["betas"]), optim_cfg["eps"])
    scaler = GradScaler(enabled=amp_enabled)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    step = 0
    total_steps = cfg["schedule"]["total_steps"]
    warmup = cfg["schedule"]["warmup_steps"]
    min_lr = cfg["schedule"]["min_lr"]
    grad_clip = float(optim_cfg.get("grad_clip_norm", 0.0))

    model.train()
    for epoch in range(cfg["train"]["max_epochs"]):
        for batch in loader:
            _maybe_print_shapes(batch, step)

            # batch["x"] is masked input, batch["y"] is full target
            # x = batch["x"].to(device)  # (B,F,T)
            # y = batch["y"].to(device)
            x = batch["x_ae"].to(device)  # masked input
            y = batch["y_spec"].to(device)  # clean target

            # U-Net expects (B, C, T) where C=F_bins
            # Ensure channel count matches config
            if x.shape[1] != F_bins:
                raise ValueError(f"Config in_channels={F_bins} but batch has F={x.shape[1]}")

            lr = cosine_schedule(step, total_steps, warmup, optim_cfg["lr"], min_lr)
            apply_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                y_hat = model(x)  # (B,F,T)
                loss, logs = loss_fn(y_hat, y)  # spectral (+ lowmag if enabled)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg["train"]["log_every"] == 0:
                print(f"[AE-pretrain] step={step} loss={loss.item():.6f} lr={lr:.2e}")

            if step % cfg["train"]["save_every"] == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


def train_pretrain_transformer_contrastive(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp_enabled = bool(cfg["train"].get("amp", False))

    mean_std = compute_or_get_stats(cfg)
    # masking at bin-level does NOT matter for transformer token masking; we just need spec
    loader = build_loader(cfg, pretrain=False, use_masking=False, mean_std=mean_std)

    enc_cfg = cfg["models"]["transformer_encoder"]
    token_cfg = cfg["patch_tokens"]

    model = SpectrogramTransformerContrastive(
        patch_f=token_cfg["patch_f"],
        patch_t=token_cfg["patch_t"],
        d_model=enc_cfg["d_model"],
        n_layers=enc_cfg["n_layers"],
        n_heads=enc_cfg["n_heads"],
        d_ff=enc_cfg["d_ff"],
        dropout=enc_cfg["dropout"],
        max_tokens=token_cfg["max_tokens"],
        proj_dim=enc_cfg.get("proj_dim", None),
    ).to(device)

    optim_cfg = cfg["optim"]
    optim = build_optimizer(optim_cfg["name"], model.parameters(), optim_cfg["lr"],
                            optim_cfg["weight_decay"], tuple(optim_cfg["betas"]), optim_cfg["eps"])
    scaler = GradScaler(enabled=amp_enabled)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    step = 0
    total_steps = cfg["schedule"]["total_steps"]
    warmup = cfg["schedule"]["warmup_steps"]
    min_lr = cfg["schedule"]["min_lr"]
    grad_clip = float(optim_cfg.get("grad_clip_norm", 0.0))

    tcfg = TokenMaskCfg(mask_ratio=token_cfg["mask_ratio"], fixed_count=token_cfg["fixed_count"])
    temperature = float(cfg["contrastive"]["temperature"])
    stop_grad = bool(cfg["contrastive"].get("stop_grad_target", True))

    model.train()
    for epoch in range(cfg["train"]["max_epochs"]):
        for batch in loader:
            x = batch["y"].to(device)  # use clean spec as input (B,F,T)

            lr = cosine_schedule(step, total_steps, warmup, optim_cfg["lr"], min_lr)
            apply_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                q, k, pmask, _meta = model(x, mask_cfg=tcfg, pretrain=True)
                if stop_grad:
                    k = k.detach()
                loss = masked_infonce_loss(q, k, pmask, temperature=temperature)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg["train"]["log_every"] == 0:
                masked = int(pmask.sum().item())
                print(f"[Tr-contrastive] step={step} loss={loss.item():.6f} masked_tokens={masked} lr={lr:.2e}")

            if step % cfg["train"]["save_every"] == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


def train_transformer_decoder_only(cfg: Dict[str, Any]):
    """
    Loads full MAE transformer, freezes encoder, trains decoder reconstruction.
    """
    device = torch.device(cfg.get("device", "cuda"))
    amp_enabled = bool(cfg["train"].get("amp", False))

    mean_std = compute_or_get_stats(cfg)
    loader = build_loader(cfg, pretrain=False, use_masking=False, mean_std=mean_std)

    enc_cfg = cfg["models"]["transformer_encoder"]
    dec_cfg = cfg["models"]["transformer_decoder"]
    token_cfg = cfg["patch_tokens"]

    model = SpectrogramMAETransformer(
        patch_f=token_cfg["patch_f"],
        patch_t=token_cfg["patch_t"],
        d_model=enc_cfg["d_model"],
        enc_layers=enc_cfg["n_layers"],
        enc_heads=enc_cfg["n_heads"],
        enc_ff=enc_cfg["d_ff"],
        dec_layers=dec_cfg["dec_layers"],
        dec_heads=dec_cfg["dec_heads"],
        dec_ff=dec_cfg["dec_ff"],
        dropout=dec_cfg["dropout"],
        max_tokens=token_cfg["max_tokens"],
        decoder_dim=dec_cfg.get("decoder_dim", None),
    ).to(device)

    # Load pretrained encoder weights if provided
    init = cfg.get("init", {})
    if init.get("load_encoder_ckpt"):
        print("[decoder-only] loading encoder ckpt:", init["load_encoder_ckpt"])
        # we load into full model with strict=False; it will fill encoder-related keys if matching
        load_checkpoint(init["load_encoder_ckpt"], model, strict=bool(init.get("strict", False)), map_location="cpu")

    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.patch_embed.parameters():
        p.requires_grad = False
    # keep decoder + recon head trainable

    # Loss for reconstruction in patch space (plus optional spectral/lowmag if you want; here patch-L2)
    # We'll still allow optional lowmag by unpatchifying (more expensive). Default: patch L2 only.
    loss_cfg = cfg["loss"]
    loss_fn = InverSynth2Loss(
        spec_w=SpectralLossWeights(**loss_cfg["spec"]),
        lowmag=LowMagConfig(**loss_cfg["lowmag"]),
        param_w=ParamLossWeights(**loss_cfg["params"]),
    ).to(device)

    optim_cfg = cfg["optim"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = build_optimizer(optim_cfg["name"], trainable, optim_cfg["lr"],
                            optim_cfg["weight_decay"], tuple(optim_cfg["betas"]), optim_cfg["eps"])
    scaler = GradScaler(enabled=amp_enabled)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    step = 0
    total_steps = cfg["schedule"]["total_steps"]
    warmup = cfg["schedule"]["warmup_steps"]
    min_lr = cfg["schedule"]["min_lr"]
    grad_clip = float(optim_cfg.get("grad_clip_norm", 0.0))

    mcfg = MaePatchMaskCfg(mask_ratio=token_cfg["mask_ratio"], fixed_count=token_cfg["fixed_count"])

    model.train()
    for epoch in range(cfg["train"]["max_epochs"]):
        for batch in loader:
            spec = batch["y"].to(device)  # clean spec

            lr = cosine_schedule(step, total_steps, warmup, optim_cfg["lr"], min_lr)
            apply_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                recon_p, target_p, pmask, meta = model(spec, mask_cfg=mcfg, pretrain=True)
                # patch loss on masked patches
                patch_loss = model.reconstruction_loss(recon_p, target_p, pmask, loss_type="l2")

                # Optional: convert patches back to spec and apply spectral/lowmag losses too
                # (costly but aligns with your paper losses)
                # If you don't want this, set loss.lowmag.enabled=false and spec alphas=0 in config.
                recon_spec = model.unpatchify(recon_p, meta) if hasattr(model, "unpatchify") else None

                if recon_spec is None:
                    loss = patch_loss
                else:
                    # If your model doesn't expose unpatchify, skip; otherwise:
                    loss_audio, _logs = loss_fn(recon_spec, spec[:, :recon_spec.shape[1], :recon_spec.shape[2]])
                    loss = patch_loss + loss_audio

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg["train"]["log_every"] == 0:
                print(f"[decoder-only] step={step} loss={loss.item():.6f} patch={patch_loss.item():.6f} lr={lr:.2e}")

            if step % cfg["train"]["save_every"] == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


def train_full_transformer(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp_enabled = bool(cfg["train"].get("amp", False))

    mean_std = compute_or_get_stats(cfg)
    loader = build_loader(cfg, pretrain=False, use_masking=False, mean_std=mean_std)

    enc_cfg = cfg["models"]["transformer_encoder"]
    dec_cfg = cfg["models"]["transformer_decoder"]
    token_cfg = cfg["patch_tokens"]

    model = SpectrogramMAETransformer(
        patch_f=token_cfg["patch_f"],
        patch_t=token_cfg["patch_t"],
        d_model=enc_cfg["d_model"],
        enc_layers=enc_cfg["n_layers"],
        enc_heads=enc_cfg["n_heads"],
        enc_ff=enc_cfg["d_ff"],
        dec_layers=dec_cfg["dec_layers"],
        dec_heads=dec_cfg["dec_heads"],
        dec_ff=dec_cfg["dec_ff"],
        dropout=dec_cfg["dropout"],
        max_tokens=token_cfg["max_tokens"],
        decoder_dim=dec_cfg.get("decoder_dim", None),
    ).to(device)

    init = cfg.get("init", {})
    if init.get("load_encoder_ckpt"):
        print("[full-transformer] loading encoder ckpt:", init["load_encoder_ckpt"])
        load_checkpoint(init["load_encoder_ckpt"], model, strict=False, map_location="cpu")
    if init.get("load_decoder_ckpt"):
        print("[full-transformer] loading decoder ckpt:", init["load_decoder_ckpt"])
        load_checkpoint(init["load_decoder_ckpt"], model, strict=False, map_location="cpu")

    loss_cfg = cfg["loss"]
    loss_fn = InverSynth2Loss(
        spec_w=SpectralLossWeights(**loss_cfg["spec"]),
        lowmag=LowMagConfig(**loss_cfg["lowmag"]),
        param_w=ParamLossWeights(**loss_cfg["params"]),
    ).to(device)

    optim_cfg = cfg["optim"]
    optim = build_optimizer(optim_cfg["name"], model.parameters(), optim_cfg["lr"],
                            optim_cfg["weight_decay"], tuple(optim_cfg["betas"]), optim_cfg["eps"])
    scaler = GradScaler(enabled=amp_enabled)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    step = 0
    total_steps = cfg["schedule"]["total_steps"]
    warmup = cfg["schedule"]["warmup_steps"]
    min_lr = cfg["schedule"]["min_lr"]
    grad_clip = float(optim_cfg.get("grad_clip_norm", 0.0))

    mcfg = MaePatchMaskCfg(mask_ratio=token_cfg["mask_ratio"], fixed_count=token_cfg["fixed_count"])

    model.train()
    for epoch in range(cfg["train"]["max_epochs"]):
        for batch in loader:
            spec = batch["y"].to(device)

            lr = cosine_schedule(step, total_steps, warmup, optim_cfg["lr"], min_lr)
            apply_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                recon_p, target_p, pmask, meta = model(spec, mask_cfg=mcfg, pretrain=True)
                patch_loss = model.reconstruction_loss(recon_p, target_p, pmask, loss_type="l2")

                # Also compute spectral + lowmag on reconstructed spectrogram (optional)
                # If you want to keep it light, set loss.spec alphas=0 and lowmag.enabled=false.
                if hasattr(model, "unpatchify"):
                    recon_spec = model.unpatchify(recon_p, meta)
                    # align sizes if cropped inside patchify
                    tgt_spec = spec[:, :recon_spec.shape[1], :recon_spec.shape[2]]
                    audio_loss, _logs = loss_fn(recon_spec, tgt_spec)
                    loss = patch_loss + audio_loss
                else:
                    loss = patch_loss

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg["train"]["log_every"] == 0:
                print(f"[full-transformer] step={step} loss={loss.item():.6f} patch={patch_loss.item():.6f} lr={lr:.2e}")

            if step % cfg["train"]["save_every"] == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


def finetune_autoencoder_nomask(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp_enabled = bool(cfg["train"].get("amp", False))

    mean_std = compute_or_get_stats(cfg)
    loader = build_loader(cfg, pretrain=False, use_masking=False, mean_std=mean_std)

    model = UNet1DTime(**cfg["models"]["autoencoder_unet"]).to(device)

    init = cfg.get("init", {})
    if init.get("load_autoencoder_ckpt"):
        print("[AE-finetune] loading AE ckpt:", init["load_autoencoder_ckpt"])
        load_checkpoint(init["load_autoencoder_ckpt"], model, strict=bool(init.get("strict", False)), map_location="cpu")

    loss_cfg = cfg["loss"]
    loss_fn = InverSynth2Loss(
        spec_w=SpectralLossWeights(**loss_cfg["spec"]),
        lowmag=LowMagConfig(**loss_cfg["lowmag"]),
        param_w=ParamLossWeights(**loss_cfg["params"]),
    ).to(device)

    optim_cfg = cfg["optim"]
    optim = build_optimizer(optim_cfg["name"], model.parameters(), optim_cfg["lr"],
                            optim_cfg["weight_decay"], tuple(optim_cfg["betas"]), optim_cfg["eps"])
    scaler = GradScaler(enabled=amp_enabled)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    step = 0
    total_steps = cfg["schedule"]["total_steps"]
    warmup = cfg["schedule"]["warmup_steps"]
    min_lr = cfg["schedule"]["min_lr"]
    grad_clip = float(optim_cfg.get("grad_clip_norm", 0.0))

    model.train()
    for epoch in range(cfg["train"]["max_epochs"]):
        for batch in loader:
            spec = batch["y"].to(device)

            lr = cosine_schedule(step, total_steps, warmup, optim_cfg["lr"], min_lr)
            apply_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp_enabled):
                y_hat = model(spec)
                loss, logs = loss_fn(y_hat, spec)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()

            if step % cfg["train"]["log_every"] == 0:
                print(f"[AE-finetune] step={step} loss={loss.item():.6f} lr={lr:.2e}")

            if step % cfg["train"]["save_every"] == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


# -----------------------------
# Main dispatch
# -----------------------------
def run(cfg: Dict[str, Any]):
    seed_all(int(cfg.get("seed", 1234)))

    exp = cfg["experiment"]  # name like "pretrain_autoencoder_mae"
    # If you used the YAML preset layout I gave earlier, you'd merge it first.
    # Here we assume cfg is already the merged config for the chosen experiment.

    print(f"== Running experiment: {cfg.get('experiment_name', exp)} ==")

    task = cfg.get("task", exp)
    if task in ["autoencoder"] and cfg.get("mode") == "pretrain":
        return train_pretrain_autoencoder(cfg)
    if task in ["transformer_encoder"] and cfg.get("mode") == "pretrain":
        return train_pretrain_transformer_contrastive(cfg)
    if task in ["transformer_decoder"]:
        return train_transformer_decoder_only(cfg)
    if task in ["full_transformer"]:
        return train_full_transformer(cfg)
    if task in ["autoencoder"] and cfg.get("mode") in ["finetune", "train"]:
        return finetune_autoencoder_nomask(cfg)

    raise ValueError(f"Unknown task/mode combo: task={task} mode={cfg.get('mode')}")


if __name__ == "__main__":
    """
    Usage:
      python train.py --config configs/exp_pretrain_autoencoder.json
      python train.py --config configs/exp_pretrain_transformer.json
      ...
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config for ONE experiment (already merged).")
    args = ap.parse_args()

    cfg = load_config_json(args.config)
    run(cfg)
