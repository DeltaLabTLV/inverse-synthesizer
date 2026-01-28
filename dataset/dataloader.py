import os
import glob
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Audio loading
# -----------------------------
def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """
    Returns:
      wav: (T,) float32
      sr
    """
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)     # (C, T)
        wav = wav.mean(dim=0)               # mono
        if target_sr is not None and sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        return wav.contiguous().float(), sr
    except Exception:
        import soundfile as sf
        wav_np, sr = sf.read(path, always_2d=False)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        wav = torch.from_numpy(wav_np).float()
        if target_sr is not None and sr != target_sr:
            wav = wav[None, None, :]
            new_len = int(round(wav.shape[-1] * (target_sr / sr)))
            wav = F.interpolate(wav, size=new_len, mode="linear", align_corners=False)[0, 0]
            sr = target_sr
        return wav.contiguous(), sr


# -----------------------------
# STFT
# -----------------------------
@dataclass
class StftConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: Optional[int] = None
    center: bool = True
    window: str = "hann"
    magnitude: bool = True
    log_mag: bool = True
    eps: float = 1e-8


def compute_stft_mag(wav: torch.Tensor, cfg: StftConfig) -> torch.Tensor:
    """
    wav: (T,)
    returns: (F, TT) float32
    """
    if cfg.win_length is None:
        cfg.win_length = cfg.n_fft

    if cfg.window != "hann":
        raise ValueError("Only hann window supported in this snippet")

    win = torch.hann_window(cfg.win_length, device=wav.device)
    X = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=win,
        center=cfg.center,
        return_complex=True,
    )
    mag = X.abs() if cfg.magnitude else X
    if cfg.magnitude and cfg.log_mag:
        mag = torch.log1p(mag)
    return mag.float()


# -----------------------------
# Bin masking (for AE pretrain only)
# -----------------------------
@dataclass
class BinMaskConfig:
    enabled: bool = True
    mask_ratio: float = 0.45
    patch_f: int = 5
    patch_t: int = 5
    mask_value: float = 0.0


def bin_mae_mask(spec: torch.Tensor, mcfg: BinMaskConfig, g: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    spec: (F, T)
    Mask 45% patches of size 5x5 (freq x time).
    Returns:
      spec_masked: (F,T)
      mask: (F,T) bool True=masked
    """
    F_bins, T_frames = spec.shape
    pf, pt = mcfg.patch_f, mcfg.patch_t

    n_pf = max(1, F_bins // pf)
    n_pt = max(1, T_frames // pt)
    total = n_pf * n_pt

    k = int(round(mcfg.mask_ratio * total))
    k = max(0, min(total, k))

    perm = torch.randperm(total, generator=g, device=spec.device)
    masked = perm[:k]

    patch_mask = torch.zeros(total, device=spec.device, dtype=torch.bool)
    patch_mask[masked] = True
    patch_mask = patch_mask.view(n_pf, n_pt)

    mask = torch.zeros((F_bins, T_frames), device=spec.device, dtype=torch.bool)
    for i in range(n_pf):
        f0, f1 = i * pf, min((i + 1) * pf, F_bins)
        for j in range(n_pt):
            if patch_mask[i, j]:
                t0, t1 = j * pt, min((j + 1) * pt, T_frames)
                mask[f0:f1, t0:t1] = True

    out = spec.clone()
    out[mask] = mcfg.mask_value
    return out, mask


# -----------------------------
# Unified Dataset
# -----------------------------
class UnifiedAudioStftDataset(Dataset):
    """
    task:
      - "ae_pretrain"          : returns masked bins + clean target
      - "ae_train"             : returns clean spec (no bin masking)
      - "tr_pretrain"          : returns clean spec (transformer masks tokens internally)
      - "full_transformer"     : returns clean spec (model may mask tokens internally)
      - "test"                 : returns clean spec (+ optional wav)
    """
    def __init__(
        self,
        files: List[str],
        task: str,
        stft_cfg: StftConfig,
        target_sr: Optional[int],
        clip_samples: Optional[int],
        mean_std: Tuple[float, float],
        bin_mask_cfg: Optional[BinMaskConfig] = None,
        return_wav: bool = False,
        seed: int = 0,
    ):
        self.files = files
        self.task = task
        self.stft_cfg = stft_cfg
        self.target_sr = target_sr
        self.clip_samples = clip_samples
        self.mean, self.std = float(mean_std[0]), float(mean_std[1])
        self.bin_mask_cfg = bin_mask_cfg or BinMaskConfig(enabled=False)
        self.return_wav = return_wav
        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def _crop_or_pad(self, wav: torch.Tensor, length: int, g: torch.Generator) -> torch.Tensor:
        T = wav.numel()
        if T == length:
            return wav
        if T > length:
            start = int(torch.randint(0, T - length + 1, (1,), generator=g).item())
            return wav[start:start + length]
        return F.pad(wav, (0, length - T))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        g = torch.Generator()
        g.manual_seed(self.base_seed + idx)

        wav, _sr = load_audio(path, self.target_sr)
        if self.clip_samples is not None:
            wav = self._crop_or_pad(wav, self.clip_samples, g)

        spec = compute_stft_mag(wav, self.stft_cfg)  # (F, T)
        spec = (spec - self.mean) / (self.std + 1e-8)  # normalize

        out: Dict[str, Any] = {"path": path, "spec": spec}

        if self.task == "ae_pretrain":
            if not self.bin_mask_cfg.enabled:
                raise ValueError("ae_pretrain requires bin_mask_cfg.enabled=True")
            spec_masked, mask = bin_mae_mask(spec, self.bin_mask_cfg, g)
            out.update({
                "x_ae": spec_masked,   # (F,T)
                "y_spec": spec,        # (F,T)
                "mask_bin": mask,      # (F,T) bool
            })

        elif self.task in ("ae_train", "tr_pretrain", "full_transformer", "test"):
            # no bin masking; transformer will mask patch tokens internally if needed
            out.update({"y_spec": spec})

        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.return_wav:
            out["wav"] = wav

        return out


# -----------------------------
# Collate: pad time to max T in batch
# -----------------------------
def collate_pad_time(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads along time axis for all (F,T) tensors to max T.
    Produces batch tensors shaped (B, F, T_max).
    """
    def pad_ft(x: torch.Tensor, T: int) -> torch.Tensor:
        if x.shape[1] == T:
            return x
        return F.pad(x, (0, T - x.shape[1]))

    # Find T_max from "spec" or "y_spec"
    T_max = 0
    for b in batch:
        base = b.get("spec", b.get("y_spec"))
        T_max = max(T_max, base.shape[1])

    paths = [b["path"] for b in batch]
    out: Dict[str, Any] = {"path": paths}

    # Stack common tensors if present
    for key in ["spec", "y_spec", "x_ae"]:
        if key in batch[0]:
            out[key] = torch.stack([pad_ft(b[key], T_max) for b in batch], dim=0)  # (B,F,T)

    if "mask_bin" in batch[0]:
        out["mask_bin"] = torch.stack([pad_ft(b["mask_bin"].to(torch.float32), T_max).bool() for b in batch], dim=0)

    if "wav" in batch[0]:
        # waveform can be variable length; here we pad it too
        wav_max = max(b["wav"].numel() for b in batch)
        def pad_1d(w: torch.Tensor, L: int) -> torch.Tensor:
            return w if w.numel() == L else F.pad(w, (0, L - w.numel()))
        out["wav"] = torch.stack([pad_1d(b["wav"], wav_max) for b in batch], dim=0)

    return out


# -----------------------------
# Helpers
# -----------------------------
def make_files(root: str, exts=("wav", "flac", "mp3", "ogg")) -> List[str]:
    files: List[str] = []
    for e in exts:
        files += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No audio files found under: {root}")
    return files


@torch.no_grad()
def compute_dataset_mean_std(
    files: List[str],
    stft_cfg: StftConfig,
    target_sr: Optional[int],
    clip_samples: Optional[int],
    max_files: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Streaming mean/std across ALL spectrogram bins (Welford).
    """
    use = files[:max_files] if max_files is not None else files
    n = 0
    mean = 0.0
    M2 = 0.0

    for p in use:
        wav, _ = load_audio(p, target_sr)
        if clip_samples is not None:
            if wav.numel() >= clip_samples:
                wav = wav[:clip_samples]
            else:
                wav = F.pad(wav, (0, clip_samples - wav.numel()))

        spec = compute_stft_mag(wav, stft_cfg).flatten()
        for v in spec:
            n += 1
            dv = float(v.item()) - mean
            mean += dv / n
            M2 += dv * (float(v.item()) - mean)

    var = M2 / max(1, n)
    std = math.sqrt(var + 1e-12)
    return float(mean), float(std)


def make_loader(
    root: str,
    task: str,
    stft_cfg: StftConfig,
    batch_size: int,
    num_workers: int,
    target_sr: Optional[int],
    clip_seconds: Optional[float],
    mean_std: Tuple[float, float],
    bin_mask_cfg: Optional[BinMaskConfig] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    return_wav: bool = False,
    seed: int = 0,
) -> DataLoader:
    files = make_files(root)
    clip_samples = int(round(target_sr * clip_seconds)) if (target_sr and clip_seconds) else None

    ds = UnifiedAudioStftDataset(
        files=files,
        task=task,
        stft_cfg=stft_cfg,
        target_sr=target_sr,
        clip_samples=clip_samples,
        mean_std=mean_std,
        bin_mask_cfg=bin_mask_cfg,
        return_wav=return_wav,
        seed=seed,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_pad_time,
    )
