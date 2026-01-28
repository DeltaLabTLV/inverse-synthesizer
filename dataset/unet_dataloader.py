import os
import glob
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Audio loading (torchaudio if available; fallback to soundfile)
# -----------------------------
def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """
    Returns:
      wav: (T,) float32 torch tensor in [-1, 1] (roughly)
      sr: sample rate
    """
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)  # (C, T)
        wav = wav.mean(dim=0)            # mono
        if target_sr is not None and sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr
        return wav.contiguous().float(), sr
    except Exception:
        # Fallback: soundfile
        import soundfile as sf
        wav_np, sr = sf.read(path, always_2d=False)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        wav = torch.from_numpy(wav_np).float()
        if target_sr is not None and sr != target_sr:
            # naive resample fallback using torch (linear). For high quality, prefer torchaudio.
            wav = wav[None, None, :]
            new_len = int(round(wav.shape[-1] * (target_sr / sr)))
            wav = F.interpolate(wav, size=new_len, mode="linear", align_corners=False)[0, 0]
            sr = target_sr
        return wav.contiguous(), sr


# -----------------------------
# STFT + feature extraction
# -----------------------------
@dataclass
class StftConfig:
    n_fft: int = 1024
    hop_length: int = 256
    win_length: Optional[int] = None
    center: bool = True
    window: str = "hann"
    magnitude: bool = True      # True -> |STFT|, False -> complex STFT
    log_mag: bool = False       # True -> log(1 + mag)
    eps: float = 1e-8


def compute_stft_feat(wav: torch.Tensor, cfg: StftConfig) -> torch.Tensor:
    """
    wav: (T,)
    returns spec: (F, TT) float32 (magnitude or log-magnitude)
    """
    if cfg.win_length is None:
        cfg.win_length = cfg.n_fft

    if cfg.window == "hann":
        win = torch.hann_window(cfg.win_length, device=wav.device)
    else:
        raise ValueError(f"Unsupported window: {cfg.window}")

    X = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=win,
        center=cfg.center,
        return_complex=True,
    )  # (F, TT) complex

    if cfg.magnitude:
        spec = X.abs()
        if cfg.log_mag:
            spec = torch.log1p(spec)
        return spec.float()
    else:
        # If you want complex, you'd return two channels etc.
        return X


# -----------------------------
# MAE-style masking (5x5 patches, 45% masked)
# -----------------------------
@dataclass
class MaskConfig:
    enable: bool = True
    mask_ratio: float = 0.45
    patch_f: int = 5
    patch_t: int = 5
    # value used for masked bins (the actual learnable token is typically in the model)
    mask_value: float = 0.0


def mae_mask_2d(spec: torch.Tensor, mcfg: MaskConfig, generator: Optional[torch.Generator] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    spec: (F, T)
    Returns:
      spec_masked: (F, T) with masked bins replaced by mcfg.mask_value
      mask: (F, T) bool tensor, True where masked
    """
    F_bins, T_frames = spec.shape
    pf, pt = mcfg.patch_f, mcfg.patch_t

    # number of patches along each axis
    n_pf = max(1, F_bins // pf)
    n_pt = max(1, T_frames // pt)
    total_patches = n_pf * n_pt

    # how many patches to mask
    num_mask = int(round(mcfg.mask_ratio * total_patches))
    num_mask = max(0, min(total_patches, num_mask))

    # sample which patches to mask
    perm = torch.randperm(total_patches, generator=generator, device=spec.device)
    masked_patch_ids = perm[:num_mask]

    # build patch mask grid (n_pf, n_pt)
    patch_mask = torch.zeros(total_patches, device=spec.device, dtype=torch.bool)
    patch_mask[masked_patch_ids] = True
    patch_mask = patch_mask.view(n_pf, n_pt)

    # expand to bin-level mask (F, T)
    mask = torch.zeros((F_bins, T_frames), device=spec.device, dtype=torch.bool)

    for i in range(n_pf):
        f0, f1 = i * pf, min((i + 1) * pf, F_bins)
        for j in range(n_pt):
            if patch_mask[i, j]:
                t0, t1 = j * pt, min((j + 1) * pt, T_frames)
                mask[f0:f1, t0:t1] = True

    spec_masked = spec.clone()
    spec_masked[mask] = mcfg.mask_value
    return spec_masked, mask


# -----------------------------
# Dataset
# -----------------------------
class AudioToStftDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        stft_cfg: StftConfig,
        mask_cfg: MaskConfig,
        pretrain: bool,
        target_sr: Optional[int] = None,
        clip_samples: Optional[int] = None,   # if set, random crop/pad waveform to this length
        mean_std: Optional[Tuple[float, float]] = None,
        seed: int = 0,
    ):
        self.files = files
        self.stft_cfg = stft_cfg
        self.mask_cfg = mask_cfg
        self.pretrain = pretrain
        self.target_sr = target_sr
        self.clip_samples = clip_samples

        # normalization stats (global)
        self.mean_std = mean_std  # (mean, std) over spectrogram values

        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def _crop_or_pad(self, wav: torch.Tensor, length: int, g: torch.Generator) -> torch.Tensor:
        T = wav.shape[0]
        if T == length:
            return wav
        if T > length:
            start = int(torch.randint(low=0, high=T - length + 1, size=(1,), generator=g).item())
            return wav[start:start + length]
        # pad
        pad = length - T
        return F.pad(wav, (0, pad))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]

        # deterministic-ish randomness per item (but changes each epoch if you change base_seed externally)
        g = torch.Generator()
        g.manual_seed(self.base_seed + idx)

        wav, sr = load_audio(path, self.target_sr)

        if self.clip_samples is not None:
            wav = self._crop_or_pad(wav, self.clip_samples, g)

        spec = compute_stft_feat(wav, self.stft_cfg)  # (F, T)

        # normalize
        if self.mean_std is not None:
            mean, std = self.mean_std
            spec_n = (spec - mean) / (std + 1e-8)
        else:
            # per-example fallback (not ideal for training consistency, but safe)
            mean = spec.mean().item()
            std = spec.std(unbiased=False).item()
            spec_n = (spec - mean) / (std + 1e-8)

        y = spec_n  # target always full spec

        if self.pretrain and self.mask_cfg.enable:
            x, mask = mae_mask_2d(spec_n, self.mask_cfg, generator=g)
        else:
            x = spec_n
            mask = torch.zeros_like(spec_n, dtype=torch.bool)

        return {
            "x": x,                 # (F, T)
            "y": y,                 # (F, T)
            "mask": mask,           # (F, T) bool
            "path": path,
        }


# -----------------------------
# Compute global mean/std (one pass)
# -----------------------------
@torch.no_grad()
def compute_dataset_mean_std(
    files: List[str],
    stft_cfg: StftConfig,
    target_sr: Optional[int] = None,
    clip_samples: Optional[int] = None,
    max_files: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Streaming mean/std over all spectrogram bins across the dataset.
    """
    n = 0
    mean = 0.0
    M2 = 0.0

    use_files = files[:max_files] if max_files is not None else files

    for p in use_files:
        wav, _ = load_audio(p, target_sr)
        if clip_samples is not None:
            if wav.numel() >= clip_samples:
                wav = wav[:clip_samples]
            else:
                wav = F.pad(wav, (0, clip_samples - wav.numel()))

        spec = compute_stft_feat(wav, stft_cfg).flatten()  # (N,)
        # Welford update
        for v in spec:
            n += 1
            dv = float(v.item()) - mean
            mean += dv / n
            M2 += dv * (float(v.item()) - mean)

    var = (M2 / max(1, n))  # population variance
    std = math.sqrt(var + 1e-12)
    return float(mean), float(std)


# -----------------------------
# Collate: pad T to max in batch (F is fixed by n_fft)
# -----------------------------
def collate_pad_time(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads along time axis to the max T in batch.
    Returns tensors shaped (B, F, T_max).
    """
    xs = [b["x"] for b in batch]      # (F, T)
    ys = [b["y"] for b in batch]
    ms = [b["mask"] for b in batch]
    paths = [b["path"] for b in batch]

    F_bins = xs[0].shape[0]
    T_max = max(x.shape[1] for x in xs)

    def pad_ft(z: torch.Tensor, T: int) -> torch.Tensor:
        if z.shape[1] == T:
            return z
        return F.pad(z, (0, T - z.shape[1]))  # pad time on the right

    x = torch.stack([pad_ft(z, T_max) for z in xs], dim=0)         # (B, F, T)
    y = torch.stack([pad_ft(z, T_max) for z in ys], dim=0)
    mask = torch.stack([pad_ft(z.to(torch.float32), T_max).bool() for z in ms], dim=0)

    return {"x": x, "y": y, "mask": mask, "path": paths}


# -----------------------------
# Helper to build file list + loader
# -----------------------------
def make_files(root: str, exts=("wav", "flac", "mp3", "ogg")) -> List[str]:
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No audio files found under: {root}")
    return files


def make_loader(
    root: str,
    pretrain: bool,
    batch_size: int = 8,
    num_workers: int = 4,
    stft_cfg: Optional[StftConfig] = None,
    mask_cfg: Optional[MaskConfig] = None,
    target_sr: Optional[int] = 44100,
    clip_seconds: Optional[float] = 2.0,
    mean_std: Optional[Tuple[float, float]] = None,
) -> DataLoader:
    stft_cfg = stft_cfg or StftConfig()
    mask_cfg = mask_cfg or MaskConfig()

    files = make_files(root)

    clip_samples = None
    if clip_seconds is not None and target_sr is not None:
        clip_samples = int(round(clip_seconds * target_sr))

    ds = AudioToStftDataset(
        files=files,
        stft_cfg=stft_cfg,
        mask_cfg=mask_cfg,
        pretrain=pretrain,
        target_sr=target_sr,
        clip_samples=clip_samples,
        mean_std=mean_std,
        seed=1234,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_pad_time,
    )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    root = "/path/to/audio_folder"

    stft_cfg = StftConfig(n_fft=1024, hop_length=256, log_mag=True)
    mask_cfg = MaskConfig(enable=True, mask_ratio=0.45, patch_f=5, patch_t=5, mask_value=0.0)

    # (Recommended) compute global stats once, then reuse:
    # mean, std = compute_dataset_mean_std(make_files(root), stft_cfg, target_sr=44100, clip_samples=int(2.0*44100), max_files=2000)
    mean, std = 0.0, 1.0  # placeholder

    loader = make_loader(
        root=root,
        pretrain=True,         # masking ON
        batch_size=4,
        stft_cfg=stft_cfg,
        mask_cfg=mask_cfg,
        target_sr=44100,
        clip_seconds=2.0,
        mean_std=(mean, std),
    )

    batch = next(iter(loader))
    print(batch["x"].shape, batch["y"].shape, batch["mask"].shape)
    # => (B, F, T) each
