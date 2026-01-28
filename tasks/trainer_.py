# train_unified.py
# Plain PyTorch trainer for:
#  1) pretrain_autoencoder (bin-masked MAE)
#  2) pretrain_transformer (contrastive)
#  3) train_transformer_decoder (freeze encoder)
#  4) train_full_transformer (encoder+decoder)
#  5) train_autoencoder_nomask (finetune/train)
#  6) test_* for each model type

import os
import math
import json
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


# -----------------------------
# Unified dataloader (CORRECT imports)
# -----------------------------
from data_stft_unified import (
    StftConfig,
    BinMaskConfig,
    make_files,
    make_loader,
    compute_dataset_mean_std,
)

# -----------------------------
# Models (adjust module names to your repo)
# -----------------------------
from unet_1d import UNet1DTime
from transformer_contrastive import (
    SpectrogramTransformerContrastive,
    PatchMaskConfig as TokenMaskConfig,
    masked_infonce_loss,
)
from full_transformer import (
    SpectrogramMAETransformer,
    PatchMaskConfig as MaePatchMaskConfig,
)

# -----------------------------
# Loss pack (spectral + lowmag + params optional)
# -----------------------------
from losses import (
    InverSynth2Loss,
    SpectralLossWeights,
    LowMagConfig,
    ParamLossWeights,
)


# ============================================================
# Utils
# ============================================================
def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_checkpoint(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer], step: int, extra: Optional[Dict[str, Any]] = None):
    ckpt = {"step": step, "model": model.state_dict()}
    if optim is not None:
        ckpt["optim"] = optim.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None, strict: bool = False, map_location: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=map_location)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print("[load] missing (first 20):", missing[:20])
    if unexpected:
        print("[load] unexpected (first 20):", unexpected[:20])
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("step", 0))


def build_optimizer(cfg: Dict[str, Any], params):
    name = cfg["name"].lower()
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 0.0))
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    eps = float(cfg.get("eps", 1e-8))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    raise ValueError(f"Unknown optimizer: {name}")


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    t = min(max(t, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def set_lr(optim: torch.optim.Optimizer, lr: float):
    for pg in optim.param_groups:
        pg["lr"] = lr


def get_global_mean_std(cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Computes global mean/std if not fixed.
    """
    norm = cfg["normalization"]
    if norm.get("use_fixed_stats", False):
        return float(norm["mean"]), float(norm["std"])

    if not norm.get("compute_if_missing", True):
        return 0.0, 1.0

    root = cfg["paths"]["data_root"]
    files = make_files(root)
    stft_cfg = StftConfig(**cfg["stft"])
    sr = cfg["audio"]["target_sr"]
    clip_seconds = cfg["audio"]["clip_seconds"]
    clip_samples = int(round(sr * clip_seconds)) if (sr and clip_seconds) else None
    max_files = norm.get("max_files_for_stats", None)
    mean, std = compute_dataset_mean_std(files, stft_cfg, sr, clip_samples, max_files=max_files)
    return float(mean), float(std)


def make_loss_pack(cfg: Dict[str, Any]) -> InverSynth2Loss:
    lc = cfg["loss"]
    return InverSynth2Loss(
        spec_w=SpectralLossWeights(**lc["spec"]),
        lowmag=LowMagConfig(**lc["lowmag"]),
        param_w=ParamLossWeights(**lc["params"]),
    )


# ============================================================
# Dataloaders for each stage (CORRECT tasks + keys)
# ============================================================
def build_stage_loader(cfg: Dict[str, Any], stage: str, mean_std: Tuple[float, float]):
    root = cfg["paths"]["data_root"]
    stft_cfg = StftConfig(**cfg["stft"])
    sr = cfg["audio"]["target_sr"]
    clip_seconds = cfg["audio"]["clip_seconds"]

    # Bin masking is ONLY used for AE pretrain
    bin_mask_cfg = BinMaskConfig(**cfg["masking"])

    if stage == "pretrain_autoencoder":
        task = "ae_pretrain"
        # ensure masking on
        bin_mask_cfg.enabled = True

    elif stage in ("train_autoencoder_nomask", "train_autoencoder", "test_autoencoder"):
        task = "ae_train" if stage != "test_autoencoder" else "test"
        bin_mask_cfg.enabled = False

    elif stage in ("pretrain_transformer", "train_transformer_decoder", "train_full_transformer", "test_transformer"):
        # transformer uses clean spec; masks tokens internally
        task = "tr_pretrain" if stage == "pretrain_transformer" else ("full_transformer" if "train_" in stage else "test")
        bin_mask_cfg.enabled = False

    else:
        raise ValueError(f"Unknown stage for loader: {stage}")

    return make_loader(
        root=root,
        task=task,
        stft_cfg=stft_cfg,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        target_sr=sr,
        clip_seconds=clip_seconds,
        mean_std=mean_std,
        bin_mask_cfg=bin_mask_cfg,
        shuffle=(stage[:4] != "test"),
        drop_last=(stage[:4] != "test"),
        return_wav=(stage[:4] == "test"),
        seed=int(cfg.get("seed", 1234)),
    )


# ============================================================
# Stage 1: pretrain autoencoder (bin-masked MAE)
# ============================================================
def run_pretrain_autoencoder(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "pretrain_autoencoder", mean_std)

    model = UNet1DTime(**cfg["models"]["autoencoder_unet"]).to(device)
    loss_fn = make_loss_pack(cfg).to(device)

    optim = build_optimizer(cfg["optim"], model.parameters())
    scaler = GradScaler(enabled=amp)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    total_steps = int(cfg["schedule"]["total_steps"])
    warmup = int(cfg["schedule"]["warmup_steps"])
    min_lr = float(cfg["schedule"]["min_lr"])
    base_lr = float(cfg["optim"]["lr"])
    clip = float(cfg["optim"].get("grad_clip_norm", 0.0))

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            # ✅ unified dataloader keys:
            x = batch["x_ae"].to(device)     # masked bins
            y = batch["y_spec"].to(device)   # clean target

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                y_hat = model(x)
                loss, _logs = loss_fn(y_hat, y)

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()

            if step % int(cfg["train"]["log_every"]) == 0:
                print(f"[AE pretrain] step={step} loss={loss.item():.6f} lr={lr:.2e}")

            if step % int(cfg["train"]["save_every"]) == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


# ============================================================
# Stage 2: pretrain transformer encoder (contrastive)
# ============================================================
def run_pretrain_transformer(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "pretrain_transformer", mean_std)

    enc_cfg = cfg["models"]["transformer_encoder"]
    tok_cfg = cfg["patch_tokens"]

    model = SpectrogramTransformerContrastive(
        patch_f=tok_cfg["patch_f"],
        patch_t=tok_cfg["patch_t"],
        d_model=enc_cfg["d_model"],
        n_layers=enc_cfg["n_layers"],
        n_heads=enc_cfg["n_heads"],
        d_ff=enc_cfg["d_ff"],
        dropout=enc_cfg["dropout"],
        max_tokens=tok_cfg["max_tokens"],
        proj_dim=enc_cfg.get("proj_dim", None),
    ).to(device)

    optim = build_optimizer(cfg["optim"], model.parameters())
    scaler = GradScaler(enabled=amp)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    tmask = TokenMaskConfig(mask_ratio=tok_cfg["mask_ratio"], fixed_count=tok_cfg["fixed_count"])
    temperature = float(cfg["contrastive"]["temperature"])
    stop_grad = bool(cfg["contrastive"].get("stop_grad_target", True))

    total_steps = int(cfg["schedule"]["total_steps"])
    warmup = int(cfg["schedule"]["warmup_steps"])
    min_lr = float(cfg["schedule"]["min_lr"])
    base_lr = float(cfg["optim"]["lr"])
    clip = float(cfg["optim"].get("grad_clip_norm", 0.0))

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            # ✅ unified keys for transformer: clean spec in y_spec
            spec = batch["y_spec"].to(device)

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                q, k, pmask, _meta = model(spec, mask_cfg=tmask, pretrain=True)
                if stop_grad:
                    k = k.detach()
                loss = masked_infonce_loss(q, k, pmask, temperature=temperature)

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()

            if step % int(cfg["train"]["log_every"]) == 0:
                print(f"[TR pretrain] step={step} loss={loss.item():.6f} masked={int(pmask.sum().item())} lr={lr:.2e}")

            if step % int(cfg["train"]["save_every"]) == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


# ============================================================
# Stage 3: train transformer decoder only (freeze encoder)
# ============================================================
def run_train_transformer_decoder(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "train_transformer_decoder", mean_std)

    enc_cfg = cfg["models"]["transformer_encoder"]
    dec_cfg = cfg["models"]["transformer_decoder"]
    tok_cfg = cfg["patch_tokens"]

    model = SpectrogramMAETransformer(
        patch_f=tok_cfg["patch_f"],
        patch_t=tok_cfg["patch_t"],
        d_model=enc_cfg["d_model"],
        enc_layers=enc_cfg["n_layers"],
        enc_heads=enc_cfg["n_heads"],
        enc_ff=enc_cfg["d_ff"],
        dec_layers=dec_cfg["dec_layers"],
        dec_heads=dec_cfg["dec_heads"],
        dec_ff=dec_cfg["dec_ff"],
        dropout=dec_cfg["dropout"],
        max_tokens=tok_cfg["max_tokens"],
        decoder_dim=dec_cfg.get("decoder_dim", None),
    ).to(device)

    # Load encoder pretrain if provided
    init = cfg.get("init", {})
    if init.get("load_encoder_ckpt"):
        print("[decoder] loading encoder ckpt:", init["load_encoder_ckpt"])
        load_checkpoint(init["load_encoder_ckpt"], model, optim=None, strict=False, map_location="cpu")

    # Freeze encoder-related params
    for p in model.encoder.parameters():
        p.requires_grad = False
    if hasattr(model, "patch_embed"):
        for p in model.patch_embed.parameters():
            p.requires_grad = False

    # Trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]

    optim = build_optimizer(cfg["optim"], trainable)
    scaler = GradScaler(enabled=amp)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    # patch masking for MAE reconstruction
    pmask_cfg = MaePatchMaskConfig(mask_ratio=tok_cfg["mask_ratio"], fixed_count=tok_cfg["fixed_count"])

    total_steps = int(cfg["schedule"]["total_steps"])
    warmup = int(cfg["schedule"]["warmup_steps"])
    min_lr = float(cfg["schedule"]["min_lr"])
    base_lr = float(cfg["optim"]["lr"])
    clip = float(cfg["optim"].get("grad_clip_norm", 0.0))

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            spec = batch["y_spec"].to(device)

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                recon_p, tgt_p, pmask, _meta = model(spec, mask_cfg=pmask_cfg, pretrain=True)
                loss = model.reconstruction_loss(recon_p, tgt_p, pmask, loss_type="l2")

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainable, clip)
            scaler.step(optim)
            scaler.update()

            if step % int(cfg["train"]["log_every"]) == 0:
                print(f"[decoder train] step={step} loss={loss.item():.6f} masked_patches={int(pmask.sum().item())} lr={lr:.2e}")

            if step % int(cfg["train"]["save_every"]) == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


# ============================================================
# Stage 4: train full transformer (encoder+decoder)
# ============================================================
def run_train_full_transformer(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "train_full_transformer", mean_std)

    enc_cfg = cfg["models"]["transformer_encoder"]
    dec_cfg = cfg["models"]["transformer_decoder"]
    tok_cfg = cfg["patch_tokens"]

    model = SpectrogramMAETransformer(
        patch_f=tok_cfg["patch_f"],
        patch_t=tok_cfg["patch_t"],
        d_model=enc_cfg["d_model"],
        enc_layers=enc_cfg["n_layers"],
        enc_heads=enc_cfg["n_heads"],
        enc_ff=enc_cfg["d_ff"],
        dec_layers=dec_cfg["dec_layers"],
        dec_heads=dec_cfg["dec_heads"],
        dec_ff=dec_cfg["dec_ff"],
        dropout=dec_cfg["dropout"],
        max_tokens=tok_cfg["max_tokens"],
        decoder_dim=dec_cfg.get("decoder_dim", None),
    ).to(device)

    init = cfg.get("init", {})
    if init.get("load_encoder_ckpt"):
        print("[full] loading encoder ckpt:", init["load_encoder_ckpt"])
        load_checkpoint(init["load_encoder_ckpt"], model, optim=None, strict=False, map_location="cpu")
    if init.get("load_decoder_ckpt"):
        print("[full] loading decoder ckpt:", init["load_decoder_ckpt"])
        load_checkpoint(init["load_decoder_ckpt"], model, optim=None, strict=False, map_location="cpu")

    optim = build_optimizer(cfg["optim"], model.parameters())
    scaler = GradScaler(enabled=amp)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    pmask_cfg = MaePatchMaskConfig(mask_ratio=tok_cfg["mask_ratio"], fixed_count=tok_cfg["fixed_count"])

    # Optional: also apply spectral/lowmag on reconstructed spectrogram if your model supports unpatchify()
    # This requires that your SpectrogramMAETransformer exposes:
    #   def unpatchify(self, patches, meta): ...
    loss_pack = make_loss_pack(cfg).to(device)

    total_steps = int(cfg["schedule"]["total_steps"])
    warmup = int(cfg["schedule"]["warmup_steps"])
    min_lr = float(cfg["schedule"]["min_lr"])
    base_lr = float(cfg["optim"]["lr"])
    clip = float(cfg["optim"].get("grad_clip_norm", 0.0))

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            spec = batch["y_spec"].to(device)

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                recon_p, tgt_p, pmask, meta = model(spec, mask_cfg=pmask_cfg, pretrain=True)
                patch_loss = model.reconstruction_loss(recon_p, tgt_p, pmask, loss_type="l2")

                # If you added model.unpatchify(), you can combine patch + spectral losses:
                if hasattr(model, "unpatchify"):
                    recon_spec = model.unpatchify(recon_p, meta)
                    tgt_spec = spec[:, :recon_spec.shape[1], :recon_spec.shape[2]]
                    audio_loss, _ = loss_pack(recon_spec, tgt_spec)
                    loss = patch_loss + audio_loss
                else:
                    loss = patch_loss

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()

            if step % int(cfg["train"]["log_every"]) == 0:
                print(f"[full train] step={step} loss={loss.item():.6f} patch={patch_loss.item():.6f} lr={lr:.2e}")

            if step % int(cfg["train"]["save_every"]) == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


# ============================================================
# Stage 5: train/finetune autoencoder without masking
# ============================================================
def run_train_autoencoder_nomask(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "train_autoencoder_nomask", mean_std)

    model = UNet1DTime(**cfg["models"]["autoencoder_unet"]).to(device)

    init = cfg.get("init", {})
    if init.get("load_autoencoder_ckpt"):
        print("[AE nomask] loading ckpt:", init["load_autoencoder_ckpt"])
        load_checkpoint(init["load_autoencoder_ckpt"], model, optim=None, strict=bool(init.get("strict", False)), map_location="cpu")

    loss_fn = make_loss_pack(cfg).to(device)
    optim = build_optimizer(cfg["optim"], model.parameters())
    scaler = GradScaler(enabled=amp)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    total_steps = int(cfg["schedule"]["total_steps"])
    warmup = int(cfg["schedule"]["warmup_steps"])
    min_lr = float(cfg["schedule"]["min_lr"])
    base_lr = float(cfg["optim"]["lr"])
    clip = float(cfg["optim"].get("grad_clip_norm", 0.0))

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            # ✅ unified keys: clean spec in y_spec
            spec = batch["y_spec"].to(device)

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                y_hat = model(spec)
                loss, _ = loss_fn(y_hat, spec)

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()

            if step % int(cfg["train"]["log_every"]) == 0:
                print(f"[AE nomask] step={step} loss={loss.item():.6f} lr={lr:.2e}")

            if step % int(cfg["train"]["save_every"]) == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                return


# ============================================================
# Tests
# ============================================================
@torch.no_grad()
def test_autoencoder(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "test_autoencoder", mean_std)

    model = UNet1DTime(**cfg["models"]["autoencoder_unet"]).to(device)
    ckpt = cfg["init"].get("load_autoencoder_ckpt")
    if not ckpt:
        raise ValueError("test_autoencoder requires init.load_autoencoder_ckpt")
    load_checkpoint(ckpt, model, strict=False, map_location="cpu")
    model.eval()

    loss_fn = make_loss_pack(cfg).to(device)
    total = 0.0
    n = 0
    for batch in loader:
        spec = batch["y_spec"].to(device)
        y_hat = model(spec)
        loss, _ = loss_fn(y_hat, spec)
        total += float(loss.item())
        n += 1
    print(f"[TEST AE] avg_loss={total/max(1,n):.6f}")


@torch.no_grad()
def test_full_transformer(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "test_transformer", mean_std)

    enc_cfg = cfg["models"]["transformer_encoder"]
    dec_cfg = cfg["models"]["transformer_decoder"]
    tok_cfg = cfg["patch_tokens"]

    model = SpectrogramMAETransformer(
        patch_f=tok_cfg["patch_f"],
        patch_t=tok_cfg["patch_t"],
        d_model=enc_cfg["d_model"],
        enc_layers=enc_cfg["n_layers"],
        enc_heads=enc_cfg["n_heads"],
        enc_ff=enc_cfg["d_ff"],
        dec_layers=dec_cfg["dec_layers"],
        dec_heads=dec_cfg["dec_heads"],
        dec_ff=dec_cfg["dec_ff"],
        dropout=dec_cfg["dropout"],
        max_tokens=tok_cfg["max_tokens"],
        decoder_dim=dec_cfg.get("decoder_dim", None),
    ).to(device)

    ckpt = cfg["init"].get("load_decoder_ckpt") or cfg["init"].get("load_encoder_ckpt")
    if not ckpt:
        raise ValueError("test_full_transformer requires init.load_*_ckpt to load a model")
    load_checkpoint(ckpt, model, strict=False, map_location="cpu")
    model.eval()

    pmask_cfg = MaePatchMaskConfig(mask_ratio=0.0, fixed_count=True)  # no masking at test; recon all
    total = 0.0
    n = 0
    for batch in loader:
        spec = batch["y_spec"].to(device)
        recon_p, tgt_p, _pmask, _meta = model(spec, mask_cfg=pmask_cfg, pretrain=False)
        loss = torch.mean((recon_p - tgt_p) ** 2)
        total += float(loss.item())
        n += 1
    print(f"[TEST TR] avg_patch_mse={total/max(1,n):.6f}")


# ============================================================
# Dispatch
# ============================================================
def run(cfg: Dict[str, Any]):
    seed_all(int(cfg.get("seed", 1234)))
    stage = cfg["stage"]  # one of the stage names below

    if stage == "pretrain_autoencoder":
        return run_pretrain_autoencoder(cfg)
    if stage == "pretrain_transformer":
        return run_pretrain_transformer(cfg)
    if stage == "train_transformer_decoder":
        return run_train_transformer_decoder(cfg)
    if stage == "train_full_transformer":
        return run_train_full_transformer(cfg)
    if stage in ("train_autoencoder_nomask", "train_autoencoder"):
        return run_train_autoencoder_nomask(cfg)
    if stage == "test_autoencoder":
        return test_autoencoder(cfg)
    if stage == "test_transformer":
        return test_full_transformer(cfg)

    raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    """
    Usage:
      python train_unified.py --config configs/exp.json

    exp.json must contain a single merged config (one stage).
    Required key: "stage"
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Provide default experiment_name if missing
    cfg.setdefault("experiment_name", cfg.get("stage", "run"))
    run(cfg)
