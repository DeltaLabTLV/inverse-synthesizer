import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = min(groups, out_ch)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def match_length_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Ensure x has length target_len on the time axis by center-crop or symmetric pad.
    x: (B, C, T)
    """
    t = x.shape[-1]
    if t == target_len:
        return x
    if t > target_len:
        start = (t - target_len) // 2
        return x[..., start:start + target_len]
    pad = target_len - t
    left = pad // 2
    right = pad - left
    return F.pad(x, (left, right))


class UNet1DTime(nn.Module):
    """
    1D U-Net over time. Frequency bins are channels.

    Input:  (B, F, T)
    Output: (B, out_channels, T)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_blocks_per_level: int = 2,
        groups: int = 8,
        dropout: float = 0.0,
        skip_scale: bool = True,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.skip_scale = skip_scale

        self.stem = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.enc_levels = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            level_out = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_blocks_per_level):
                blocks.append(ConvBlock1D(ch, level_out, groups=groups, dropout=dropout))
                ch = level_out
            self.enc_levels.append(blocks)
            self.downs.append(Downsample1D(ch) if i < len(channel_mults) - 1 else nn.Identity())

        # Bottleneck
        self.mid = nn.Sequential(
            ConvBlock1D(ch, ch, groups=groups, dropout=dropout),
            ConvBlock1D(ch, ch, groups=groups, dropout=dropout),
        )

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_levels = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            level_out = base_channels * mult

            self.ups.append(Upsample1D(ch) if i < len(channel_mults) - 1 else nn.Identity())

            blocks = nn.ModuleList()
            blocks.append(
                ConvBlock1D(ch + level_out, level_out)
            )
            ch = level_out

            # remaining blocks: NO skip concat
            for _ in range(num_blocks_per_level - 1):
                blocks.append(
                    ConvBlock1D(ch, level_out)
                )
                ch = level_out
            # for _ in range(num_blocks_per_level):
            #     # concat skip => in channels increase by skip channels (which equals level_out)
            #     blocks.append(ConvBlock1D(ch + level_out, level_out, groups=groups, dropout=dropout))
            #     ch = level_out
            self.dec_levels.append(blocks)

        self.out = nn.Sequential(
            nn.GroupNorm(min(groups, ch), ch),
            nn.SiLU(),
            nn.Conv1d(ch, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, T)
        """
        h = self.stem(x)

        skips: List[torch.Tensor] = []

        # Encode
        for blocks, down in zip(self.enc_levels, self.downs):
            for block in blocks:
                h = block(h)
            skips.append(h)     # save per-level skip (common in U-Nets)
            h = down(h)

        # Bottleneck
        h = self.mid(h)

        # Decode (reverse order)
        for up, blocks in zip(self.ups, self.dec_levels):
            h = up(h)
            skip = skips.pop()

            # handle odd lengths / mismatch
            h = match_length_1d(h, skip.shape[-1])

            # concat along channels (freq/feature channels)
            h = torch.cat([h, skip], dim=1)

            # optional scaling helps keep variance stable
            if self.skip_scale:
                h = h * (2 ** -0.5)

            for block in blocks:
                h = block(h)

        return self.out(h)


# -----------------------------
# Quick sanity check
# -----------------------------
if __name__ == "__main__":
    B, C, T = 4, 256, 513  # odd T to test length matching
    x = torch.randn(B, C, T)

    net = UNet1DTime(
        in_channels=C,
        out_channels=C,          # typical: predict same shape (e.g., noise/residual)
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_blocks_per_level=2,
        dropout=0.0,
    )

    y = net(x)
    print("in:", x.shape, "out:", y.shape)
