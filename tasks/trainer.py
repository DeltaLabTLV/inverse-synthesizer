# train_unified_tb.py
# Plain PyTorch trainer WITH:
#  ✅ unified dataloader imports (data_stft_unified.py)
#  ✅ TensorBoard logging (loss, lr, param metrics)
#  ✅ optional parameter supervision: accuracy + MSE if dataloader provides GT params
#
# Supports stages:ssdaaaaaawc vv
#   - train_full_transformer
#   - train_autoencoder_nomask
#   - test_autoencoder
#   - test_transformer
#
# IMPORTANT (for params):
# Your dataloader batches should optionally include:
#   - batch["tgt_cont"] : (B, P_cont) float
#   - batch["tgt_cat"]  : List[Tensor] of length P_cat, each (B,) long
#     OR alternatively batch["tgt_cat"] can be a dict {name: Tensor(B,)}
#
# Your model forward can optionally return one of:
#   A) y_hat_spec (Tensor: B,F,T)   (no params)
#   B) (y_hat_spec, pred_cont, pred_cat_logits)
#   C) dict with keys:
#        "spec": Tensor(B,F,T)
#        "pred_cont": Tensor(B,P_cont) or None
#        "pred_cat_logits": List[Tensor(B,C_i)] or dict {name: Tensor(B,C)}
#
# If your model does not predict params, param logging becomes "N/A" automatically.

import os
import math
import json
from typing import Any, Dict, Optional, Tuple, List, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# Unified dataloader (correct imports)
# -----------------------------
from data_stft_unified import (
    StftConfig,
    BinMaskConfig,
    make_files,
    make_loader,
    compute_dataset_mean_std,
)

# -----------------------------
# Models (adjust to your repo)
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


def make_tb_writer(cfg: Dict[str, Any]) -> SummaryWriter:
    log_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"], "tb")
    ensure_dir(log_dir)
    return SummaryWriter(log_dir=log_dir)


def save_checkpoint(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer], step: int):
    ckpt = {"step": step, "model": model.state_dict()}
    if optim is not None:
        ckpt["optim"] = optim.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    strict: bool = False,
    map_location: str = "cpu",
) -> int:
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
# Param metrics (accuracy + mse) - optional
# ============================================================
def _to_list_cat(x: Union[List[torch.Tensor], Dict[str, torch.Tensor]]) -> Tuple[List[str], List[torch.Tensor]]:
    if isinstance(x, dict):
        names = list(x.keys())
        return names, [x[k] for k in names]
    return [f"cat_{i}" for i in range(len(x))], list(x)


@torch.no_grad()
def compute_param_metrics(
    pred_cont: Optional[torch.Tensor],
    tgt_cont: Optional[torch.Tensor],
    pred_cat_logits: Optional[Union[List[torch.Tensor], Dict[str, torch.Tensor]]],
    tgt_cat: Optional[Union[List[torch.Tensor], Dict[str, torch.Tensor]]],
) -> Dict[str, float]:
    """
    Returns scalar metrics dict (averaged over batch):
      - param/cont_mse
      - param/cat_acc_mean
      - param/cat_acc/<name>
    """
    metrics: Dict[str, float] = {}

    if pred_cont is not None and tgt_cont is not None and pred_cont.numel() > 0:
        mse = torch.mean((pred_cont - tgt_cont) ** 2).item()
        metrics["param/cont_mse"] = float(mse)

    if pred_cat_logits is not None and tgt_cat is not None:
        logit_names, logits_list = _to_list_cat(pred_cat_logits)
        tgt_names, tgt_list = _to_list_cat(tgt_cat)

        # align by order; if dicts are used, ensure same keys
        if (isinstance(pred_cat_logits, dict) and isinstance(tgt_cat, dict)) and (set(logit_names) != set(tgt_names)):
            # can't align reliably
            return metrics

        accs = []
        for name, logits, tgt in zip(logit_names, logits_list, tgt_list):
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == tgt).float().mean().item()
            metrics[f"param/cat_acc/{name}"] = float(acc)
            accs.append(acc)

        if len(accs) > 0:
            metrics["param/cat_acc_mean"] = float(sum(accs) / len(accs))

    return metrics


# ============================================================
# Output unpacker (so trainer works with models w/ or w/o param heads)
# ============================================================
def unpack_model_output(out: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
    """
    Returns: (spec_hat, pred_cont, pred_cat_logits)
    pred_cat_logits may be List[Tensor] or dict.
    """
    if isinstance(out, torch.Tensor):
        return out, None, None
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            return out[0], out[1], out[2]
        if len(out) == 2:
            return out[0], out[1], None
        raise ValueError(f"Unexpected tuple/list output length: {len(out)}")
    if isinstance(out, dict):
        spec_hat = out.get("spec")
        if spec_hat is None:
            raise ValueError("Dict output must contain key 'spec'")
        return spec_hat, out.get("pred_cont", None), out.get("pred_cat_logits", None)

    raise ValueError(f"Unsupported model output type: {type(out)}")


# ============================================================
# Dataloaders per stage
# ============================================================
def build_stage_loader(cfg: Dict[str, Any], stage: str, mean_std: Tuple[float, float]):
    root = cfg["paths"]["data_root"]
    stft_cfg = StftConfig(**cfg["stft"])
    sr = cfg["audio"]["target_sr"]
    clip_seconds = cfg["audio"]["clip_seconds"]

    bin_mask_cfg = BinMaskConfig(**cfg["masking"])

    if stage == "pretrain_autoencoder":
        task = "ae_pretrain"
        bin_mask_cfg.enabled = True
    elif stage in ("train_autoencoder_nomask", "train_autoencoder", "test_autoencoder"):
        task = "ae_train" if stage != "test_autoencoder" else "test"
        bin_mask_cfg.enabled = False
    elif stage in ("pretrain_transformer", "train_transformer_decoder", "train_full_transformer", "test_transformer"):
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
        shuffle=(not stage.startswith("test")),
        drop_last=(not stage.startswith("test")),
        return_wav=stage.startswith("test"),
        seed=int(cfg.get("seed", 1234)),
    )


# ============================================================
# Stage 1: pretrain autoencoder (bin-masked MAE)
# ============================================================
def run_pretrain_autoencoder(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    writer = make_tb_writer(cfg)
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
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            x = batch["x_ae"].to(device)     # masked bins
            y = batch["y_spec"].to(device)   # clean target

            # Optional GT params
            tgt_cont = batch.get("tgt_cont")
            tgt_cat = batch.get("tgt_cat")
            if tgt_cont is not None:
                tgt_cont = tgt_cont.to(device)
            if isinstance(tgt_cat, list):
                tgt_cat = [t.to(device) for t in tgt_cat]
            elif isinstance(tgt_cat, dict):
                tgt_cat = {k: v.to(device) for k, v in tgt_cat.items()}

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                out = model(x)
                y_hat, pred_cont, pred_cat_logits = unpack_model_output(out)
                loss, logs = loss_fn(
                    y_hat, y,
                    pred_cont=pred_cont,
                    tgt_cont=tgt_cont,
                    pred_cat_logits=pred_cat_logits,
                    tgt_cat=tgt_cat,
                )

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()

            # ---- TensorBoard ----
            if step % log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                for k, v in logs.items():
                    # logs from InverSynth2Loss are floats
                    if isinstance(v, (float, int)):
                        writer.add_scalar(f"train/{k}", float(v), step)

                # optional param metrics (accuracy/mse) if model outputs + GT available
                pm = compute_param_metrics(pred_cont, tgt_cont, pred_cat_logits, tgt_cat)
                for k, v in pm.items():
                    writer.add_scalar(f"train/{k}", v, step)

                print(f"[AE pretrain] step={step} loss={loss.item():.6f} lr={lr:.2e}")

            if step % save_every == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                writer.close()
                return


# ============================================================
# Stage 2: pretrain transformer encoder (contrastive)
# ============================================================
def run_pretrain_transformer(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    writer = make_tb_writer(cfg)
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
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
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

            if step % log_every == 0:
                writer.add_scalar("train/contrastive_loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/masked_tokens", float(pmask.sum().item()), step)
                print(f"[TR pretrain] step={step} loss={loss.item():.6f} masked={int(pmask.sum().item())} lr={lr:.2e}")

            if step % save_every == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                writer.close()
                return


# ============================================================
# Stage 3: train transformer decoder only (freeze encoder)
# ============================================================
def run_train_transformer_decoder(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    writer = make_tb_writer(cfg)
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

    init = cfg.get("init", {})
    if init.get("load_encoder_ckpt"):
        print("[decoder] loading encoder ckpt:", init["load_encoder_ckpt"])
        load_checkpoint(init["load_encoder_ckpt"], model, optim=None, strict=False, map_location="cpu")

    # Freeze encoder-related
    for p in model.encoder.parameters():
        p.requires_grad = False
    if hasattr(model, "patch_embed"):
        for p in model.patch_embed.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = build_optimizer(cfg["optim"], trainable)
    scaler = GradScaler(enabled=amp)

    out_dir = os.path.join(cfg["paths"]["out_dir"], cfg["experiment_name"])
    ensure_dir(out_dir)

    pmask_cfg = MaePatchMaskConfig(mask_ratio=tok_cfg["mask_ratio"], fixed_count=tok_cfg["fixed_count"])

    total_steps = int(cfg["schedule"]["total_steps"])
    warmup = int(cfg["schedule"]["warmup_steps"])
    min_lr = float(cfg["schedule"]["min_lr"])
    base_lr = float(cfg["optim"]["lr"])
    clip = float(cfg["optim"].get("grad_clip_norm", 0.0))
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

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

            if step % log_every == 0:
                writer.add_scalar("train/patch_loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/masked_patches", float(pmask.sum().item()), step)
                print(f"[decoder train] step={step} patch_loss={loss.item():.6f} lr={lr:.2e}")

            if step % save_every == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                writer.close()
                return


# ============================================================
# Stage 4: train full transformer (encoder+decoder)
# ============================================================
def run_train_full_transformer(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    writer = make_tb_writer(cfg)
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
    loss_pack = make_loss_pack(cfg).to(device)

    total_steps = int(cfg["schedule"]["total_steps"])
    warmup = int(cfg["schedule"]["warmup_steps"])
    min_lr = float(cfg["schedule"]["min_lr"])
    base_lr = float(cfg["optim"]["lr"])
    clip = float(cfg["optim"].get("grad_clip_norm", 0.0))
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            spec = batch["y_spec"].to(device)

            # Optional GT params
            tgt_cont = batch.get("tgt_cont")
            tgt_cat = batch.get("tgt_cat")
            if tgt_cont is not None:
                tgt_cont = tgt_cont.to(device)
            if isinstance(tgt_cat, list):
                tgt_cat = [t.to(device) for t in tgt_cat]
            elif isinstance(tgt_cat, dict):
                tgt_cat = {k: v.to(device) for k, v in tgt_cat.items()}

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                out = model(spec, mask_cfg=pmask_cfg, pretrain=True)
                # SpectrogramMAETransformer forward returns recon_patches; not spec_hat.
                recon_p, tgt_p, pmask, meta = out
                patch_loss = model.reconstruction_loss(recon_p, tgt_p, pmask, loss_type="l2")

                # If your model has unpatchify() we can also do spectral/lowmag on spectrogram
                if hasattr(model, "unpatchify"):
                    recon_spec = model.unpatchify(recon_p, meta)
                    tgt_spec = spec[:, :recon_spec.shape[1], :recon_spec.shape[2]]
                    # no params in this model by default; if you added param heads, call loss_pack with them
                    loss_audio, logs = loss_pack(recon_spec, tgt_spec)
                    loss = patch_loss + loss_audio
                else:
                    logs = {}
                    loss = patch_loss

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()

            if step % log_every == 0:
                writer.add_scalar("train/total_loss", loss.item(), step)
                writer.add_scalar("train/patch_loss", patch_loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/masked_patches", float(pmask.sum().item()), step)

                for k, v in logs.items():
                    if isinstance(v, (float, int)):
                        writer.add_scalar(f"train/{k}", float(v), step)

                # If you added param heads to the transformer and return them from forward,
                # compute metrics here similarly (pred_cont/pred_cat_logits).
                # (By default, pred_cont=None, pred_cat_logits=None)
                print(f"[full train] step={step} total={loss.item():.6f} patch={patch_loss.item():.6f} lr={lr:.2e}")

            if step % save_every == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                writer.close()
                return


# ============================================================
# Stage 5: train/finetune autoencoder without masking
# ============================================================
def run_train_autoencoder_nomask(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    amp = bool(cfg["train"].get("amp", False))

    writer = make_tb_writer(cfg)
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
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])

    step = 0
    model.train()
    for epoch in range(int(cfg["train"]["max_epochs"])):
        for batch in loader:
            spec = batch["y_spec"].to(device)

            # Optional GT params
            tgt_cont = batch.get("tgt_cont")
            tgt_cat = batch.get("tgt_cat")
            if tgt_cont is not None:
                tgt_cont = tgt_cont.to(device)
            if isinstance(tgt_cat, list):
                tgt_cat = [t.to(device) for t in tgt_cat]
            elif isinstance(tgt_cat, dict):
                tgt_cat = {k: v.to(device) for k, v in tgt_cat.items()}

            lr = cosine_lr(step, total_steps, warmup, base_lr, min_lr)
            set_lr(optim, lr)

            optim.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                out = model(spec)
                y_hat, pred_cont, pred_cat_logits = unpack_model_output(out)
                loss, logs = loss_fn(
                    y_hat, spec,
                    pred_cont=pred_cont,
                    tgt_cont=tgt_cont,
                    pred_cat_logits=pred_cat_logits,
                    tgt_cat=tgt_cat,
                )

            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()

            if step % log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                for k, v in logs.items():
                    if isinstance(v, (float, int)):
                        writer.add_scalar(f"train/{k}", float(v), step)

                pm = compute_param_metrics(pred_cont, tgt_cont, pred_cat_logits, tgt_cat)
                for k, v in pm.items():
                    writer.add_scalar(f"train/{k}", v, step)

                print(f"[AE nomask] step={step} loss={loss.item():.6f} lr={lr:.2e}")

            if step % save_every == 0 and step > 0:
                save_checkpoint(os.path.join(out_dir, f"ckpt_step{step}.pt"), model, optim, step)

            step += 1
            if step >= total_steps:
                save_checkpoint(os.path.join(out_dir, "final.pt"), model, optim, step)
                writer.close()
                return


# ============================================================
# Tests (logged to TensorBoard)
# ============================================================
@torch.no_grad()
def test_autoencoder(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    writer = make_tb_writer(cfg)

    mean_std = get_global_mean_std(cfg)
    loader = build_stage_loader(cfg, "test_autoencoder", mean_std)

    model = UNet1DTime(**cfg["models"]["autoencoder_unet"]).to(device)
    ckpt = cfg.get("init", {}).get("load_autoencoder_ckpt")
    if not ckpt:
        raise ValueError("test_autoencoder requires init.load_autoencoder_ckpt")
    load_checkpoint(ckpt, model, strict=False, map_location="cpu")
    model.eval()

    loss_fn = make_loss_pack(cfg).to(device)

    total_loss = 0.0
    n = 0
    # param metric accumulators
    cont_mse_sum = 0.0
    cont_mse_n = 0
    cat_acc_sums: Dict[str, float] = {}
    cat_acc_ns: Dict[str, int] = {}

    for batch in loader:
        spec = batch["y_spec"].to(device)

        tgt_cont = batch.get("tgt_cont")
        tgt_cat = batch.get("tgt_cat")
        if tgt_cont is not None:
            tgt_cont = tgt_cont.to(device)
        if isinstance(tgt_cat, list):
            tgt_cat = [t.to(device) for t in tgt_cat]
        elif isinstance(tgt_cat, dict):
            tgt_cat = {k: v.to(device) for k, v in tgt_cat.items()}

        out = model(spec)
        y_hat, pred_cont, pred_cat_logits = unpack_model_output(out)

        loss, _logs = loss_fn(
            y_hat, spec,
            pred_cont=pred_cont,
            tgt_cont=tgt_cont,
            pred_cat_logits=pred_cat_logits,
            tgt_cat=tgt_cat,
        )

        total_loss += float(loss.item())
        n += 1

        pm = compute_param_metrics(pred_cont, tgt_cont, pred_cat_logits, tgt_cat)
        if "param/cont_mse" in pm:
            cont_mse_sum += pm["param/cont_mse"]
            cont_mse_n += 1
        for k, v in pm.items():
            if k.startswith("param/cat_acc/"):
                cat_acc_sums[k] = cat_acc_sums.get(k, 0.0) + float(v)
                cat_acc_ns[k] = cat_acc_ns.get(k, 0) + 1

    avg_loss = total_loss / max(1, n)
    writer.add_scalar("test/avg_loss", avg_loss, 0)

    if cont_mse_n > 0:
        writer.add_scalar("test/param/cont_mse", cont_mse_sum / cont_mse_n, 0)

    for k in cat_acc_sums:
        writer.add_scalar(f"test/{k}", cat_acc_sums[k] / max(1, cat_acc_ns[k]), 0)

    writer.close()
    print(f"[TEST AE] avg_loss={avg_loss:.6f}")


@torch.no_grad()
def test_transformer(cfg: Dict[str, Any]):
    device = torch.device(cfg.get("device", "cuda"))
    writer = make_tb_writer(cfg)

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

    ckpt = cfg.get("init", {}).get("load_decoder_ckpt") or cfg.get("init", {}).get("load_encoder_ckpt")
    if not ckpt:
        raise ValueError("test_transformer requires init.load_decoder_ckpt (preferred) or init.load_encoder_ckpt")
    load_checkpoint(ckpt, model, strict=False, map_location="cpu")
    model.eval()

    # No masking in test:
    pmask_cfg = MaePatchMaskConfig(mask_ratio=0.0, fixed_count=True)

    total_patch_mse = 0.0
    n = 0
    for batch in loader:
        spec = batch["y_spec"].to(device)
        recon_p, tgt_p, _pmask, _meta = model(spec, mask_cfg=pmask_cfg, pretrain=False)
        mse = torch.mean((recon_p - tgt_p) ** 2).item()
        total_patch_mse += float(mse)
        n += 1

    avg = total_patch_mse / max(1, n)
    writer.add_scalar("test/avg_patch_mse", avg, 0)
    writer.close()
    print(f"[TEST TR] avg_patch_mse={avg:.6f}")


# ============================================================
# Dispatch
# ============================================================
def run(cfg: Dict[str, Any]):
    seed_all(int(cfg.get("seed", 1234)))
    stage = cfg["stage"]
    cfg.setdefault("experiment_name", stage)

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
        return test_transformer(cfg)

    raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    """
    Usage:
      python train_unified_tb.py --config path/to/exp.json

    exp.json must contain a single merged config + "stage".
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    run(cfg)
