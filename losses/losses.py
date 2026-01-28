# losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Spectral losses
# -----------------------------
def spectral_convergence(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SC = ||x_hat - x||_F / (||x||_F + eps)
    x_hat, x: (B, F, T) or any shape (batch first recommended)
    """
    diff = x_hat - x
    num = torch.linalg.norm(diff.reshape(diff.shape[0], -1), ord=2, dim=1)
    den = torch.linalg.norm(x.reshape(x.shape[0], -1), ord=2, dim=1).clamp_min(eps)
    return (num / den).mean()


def log_mag_l2(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean squared error in log-magnitude domain:
      mean( (log(x_hat+eps) - log(x+eps))^2 )
    Good proxy for "log spectral distance" components.
    """
    return F.mse_loss(torch.log(x_hat.clamp_min(eps)), torch.log(x.clamp_min(eps)))


def l1_mag(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x_hat, x)


def l2_mag(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


# -----------------------------
# Your low-magnitude weighted loss
# -----------------------------
def lowmag_weighted_mse(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    eps: float = 1e-4,
    detach_weight: bool = True,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    L_weighted = sum w(x) * (x_hat - x)^2,  w(x)=1/(x+eps)
    x_hat, x: (B,F,T)

    detach_weight=True is usually more stable (prevents weight exploding dynamics).
    """
    w = 1.0 / (x + eps)
    if detach_weight:
        w = w.detach()

    loss = w * (x_hat - x) ** 2
    if reduce == "mean":
        return loss.mean()
    if reduce == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduce: {reduce}")


# -----------------------------
# Parameter losses (InverSynth II style)
# -----------------------------
def continuous_param_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, P_cont)
    """
    return F.mse_loss(pred, target)


def categorical_param_ce(
    logits_list: List[torch.Tensor],
    target_list: List[torch.Tensor],
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Each categorical parameter can have different num_classes.
    logits_list: list of (B, C_i)
    target_list: list of (B,) long in [0..C_i-1]
    """
    if len(logits_list) == 0:
        return torch.tensor(0.0, device=target_list[0].device if target_list else "cpu")

    losses = []
    for logits, tgt in zip(logits_list, target_list):
        losses.append(
            F.cross_entropy(logits, tgt.long(), label_smoothing=label_smoothing)
        )
    return torch.stack(losses).mean()


# -----------------------------
# Combined config + main Loss module
# -----------------------------
@dataclass
class SpectralLossWeights:
    # "spec term" (your Eq: alpha1*L1 + alpha2*L2^2)
    alpha_l1: float = 1.0
    alpha_l2: float = 0.0

    # optional extra spectral terms
    alpha_sc: float = 0.0
    alpha_log: float = 0.0


@dataclass
class LowMagConfig:
    enabled: bool = False
    eps: float = 1e-4
    detach_weight: bool = True
    beta: float = 0.7  # L_total = beta*L_spec + (1-beta)*L_weighted


@dataclass
class ParamLossWeights:
    # InverSynth II describes Lp as avg(CE for cat, L2 for cont) :contentReference[oaicite:2]{index=2}
    enabled: bool = True
    weight: float = 1.0
    label_smoothing: float = 0.0


class InverSynth2Loss(nn.Module):
    """
    A practical "loss pack" that you can use for:
      - spectrogram reconstruction loss (proxy / MAE / decoder recon)
      - parameter prediction loss (continuous + categorical)
      - optional low-mag perceptual weighting

    Expected spectrograms:
      x_hat, x: (B, F, T) non-negative magnitude (or log-magnitude but then disable log term accordingly)
    """

    def __init__(
        self,
        spec_w: SpectralLossWeights = SpectralLossWeights(),
        lowmag: LowMagConfig = LowMagConfig(),
        param_w: ParamLossWeights = ParamLossWeights(),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.spec_w = spec_w
        self.lowmag = lowmag
        self.param_w = param_w
        self.eps = eps

    def spectral_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # base spec combination (your Eq)
        L_spec = 0.0
        logs: Dict[str, float] = {}

        if self.spec_w.alpha_l1 != 0.0:
            l = l1_mag(x_hat, x)
            L_spec = L_spec + self.spec_w.alpha_l1 * l
            logs["spec_l1"] = float(l.detach().cpu())

        if self.spec_w.alpha_l2 != 0.0:
            l = l2_mag(x_hat, x)
            L_spec = L_spec + self.spec_w.alpha_l2 * l
            logs["spec_l2"] = float(l.detach().cpu())

        if self.spec_w.alpha_sc != 0.0:
            l = spectral_convergence(x_hat, x, eps=self.eps)
            L_spec = L_spec + self.spec_w.alpha_sc * l
            logs["spec_sc"] = float(l.detach().cpu())

        if self.spec_w.alpha_log != 0.0:
            l = log_mag_l2(x_hat, x, eps=self.eps)
            L_spec = L_spec + self.spec_w.alpha_log * l
            logs["spec_log"] = float(l.detach().cpu())

        logs["L_spec"] = float(torch.as_tensor(L_spec).detach().cpu())
        return torch.as_tensor(L_spec, device=x_hat.device), logs

    def lowmag_total(self, x_hat: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        L_total = beta * L_spec + (1-beta) * L_weighted
        """
        L_spec, logs = self.spectral_loss(x_hat, x)

        if not self.lowmag.enabled:
            logs["L_total_spec_only"] = float(L_spec.detach().cpu())
            return L_spec, logs

        L_w = lowmag_weighted_mse(
            x_hat, x,
            eps=self.lowmag.eps,
            detach_weight=self.lowmag.detach_weight,
            reduce="mean",
        )
        beta = float(self.lowmag.beta)
        L_total = beta * L_spec + (1.0 - beta) * L_w

        logs["lowmag_weighted"] = float(L_w.detach().cpu())
        logs["beta"] = beta
        logs["L_total"] = float(L_total.detach().cpu())
        return L_total, logs

    def param_loss(
        self,
        pred_cont: Optional[torch.Tensor],
        tgt_cont: Optional[torch.Tensor],
        pred_cat_logits: Optional[List[torch.Tensor]],
        tgt_cat: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
          Lp = 0.5*(CE_cat + MSE_cont) (if both exist), otherwise the one that exists.
        """
        device = None
        if pred_cont is not None:
            device = pred_cont.device
        elif pred_cat_logits and len(pred_cat_logits) > 0:
            device = pred_cat_logits[0].device
        else:
            device = "cpu"

        logs: Dict[str, float] = {}
        parts = []

        if pred_cont is not None and tgt_cont is not None and pred_cont.numel() > 0:
            l_cont = continuous_param_mse(pred_cont, tgt_cont)
            parts.append(l_cont)
            logs["param_cont_mse"] = float(l_cont.detach().cpu())

        if pred_cat_logits is not None and tgt_cat is not None and len(pred_cat_logits) > 0:
            l_cat = categorical_param_ce(
                pred_cat_logits, tgt_cat,
                label_smoothing=self.param_w.label_smoothing,
            )
            parts.append(l_cat)
            logs["param_cat_ce"] = float(l_cat.detach().cpu())

        if len(parts) == 0:
            return torch.tensor(0.0, device=device), {"Lp": 0.0}

        if len(parts) == 1:
            Lp = parts[0]
        else:
            Lp = 0.5 * (parts[0] + parts[1])  # avg(cat, cont) :contentReference[oaicite:3]{index=3}

        logs["Lp"] = float(Lp.detach().cpu())
        return Lp, logs

    def forward(
        self,
        x_hat_spec: torch.Tensor,
        x_spec: torch.Tensor,
        # params (optional)
        pred_cont: Optional[torch.Tensor] = None,
        tgt_cont: Optional[torch.Tensor] = None,
        pred_cat_logits: Optional[List[torch.Tensor]] = None,
        tgt_cat: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
          total_loss, logs
        """
        # 1) spectral / low-mag combo
        L_audio, logs = self.lowmag_total(x_hat_spec, x_spec)

        # 2) parameter loss (optional)
        if self.param_w.enabled:
            Lp, p_logs = self.param_loss(pred_cont, tgt_cont, pred_cat_logits, tgt_cat)
            logs.update(p_logs)
            total = L_audio + self.param_w.weight * Lp
            logs["L_audio"] = float(L_audio.detach().cpu())
            logs["L_total_all"] = float(total.detach().cpu())
            return total, logs

        logs["L_audio"] = float(L_audio.detach().cpu())
        logs["L_total_all"] = float(L_audio.detach().cpu())
        return L_audio, logs
