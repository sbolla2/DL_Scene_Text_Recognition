"""
TSRN — Text Super-Resolution Network.

Faithful PyTorch implementation of the TSRN from:
  Wang et al., "Scene Text Image Super-Resolution in the Wild", ECCV 2020.

Used as the SR backbone inside TPGSR. Key properties:
  - Input: (B, C_in, 16, 64) where C_in is 3 (RGB) or 4 (RGB + mask)
  - Output: (B, 3, 32, 128) RGB at 2x scale
  - Core block: recurrent residual block with a horizontal BiLSTM mixed in,
    giving the network sequence-aware features (important for text)
  - Optional STN (spatial transformer) at the front to rectify slightly
    skewed/rotated input before SR

The TPGSR paper adds a prior-injection step: text priors (T, num_classes)
are projected to a spatial feature map and concatenated with an intermediate
TSRN feature map. That fusion happens inside the TPGSR wrapper (tpgsr_model.py),
not here. This file is pure TSRN — no prior-awareness baked in.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Optional Spatial Transformer Network (STN) for front-end rectification
# ---------------------------------------------------------------------------
class STN(nn.Module):
    """Tiny STN that predicts an affine transform and resamples the input.

    Useful when the LR input has some rotation/skew; the STN learns to
    undo it before the SR body sees the image.
    """

    def __init__(self, in_channels):
        super().__init__()
        # For low-resolution inputs (16x64), use padded convs and smaller kernels
        # so spatial dims don't collapse. Original paper used 28x28-ish inputs.
        self.loc = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),   # preserve size
            nn.MaxPool2d(2, 2),                                     # 16x64 -> 8x32
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),            # preserve size
            nn.MaxPool2d(2, 2),                                     # 8x32 -> 4x16
            nn.ReLU(True),
        )
        # Lazily compute the flatten size on the first forward pass.
        self._flatten_size = None
        self.fc = None

        # Identity init for the final affine — STN starts as a no-op
        self._identity_initialized = False

    def _build_fc(self, flatten_size):
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3),
        ).to(next(self.loc.parameters()).device)

        # Init the last linear so the predicted transform is identity
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )
        self._identity_initialized = True

    def forward(self, x):
        feats = self.loc(x)
        b = feats.size(0)
        flat = feats.view(b, -1)

        if self.fc is None:
            self._build_fc(flat.size(1))

        theta = self.fc(flat).view(b, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True, padding_mode='border')


# ---------------------------------------------------------------------------
# Core TSRN block: Conv -> BatchNorm -> PReLU -> Conv -> BatchNorm + BiLSTM
# ---------------------------------------------------------------------------
class RecurrentResidualBlock(nn.Module):
    """The 'recurrent' part: a horizontal BiLSTM inside a residual conv block.

    Designed for text images — letters have strong horizontal structure, so
    running an LSTM along the width dimension helps propagate character-shape
    information across the image.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Horizontal BiLSTM: treat each row of the feature map as a sequence.
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=channels // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Horizontal BiLSTM:
        # (B, C, H, W) -> treat each of the H rows as an independent sequence of W steps.
        b, c, h, w = out.shape
        # (B, H, W, C)
        seq = out.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
        seq, _ = self.lstm(seq)              # (B*H, W, C)
        seq = seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        out = out + seq

        out = out + residual
        return out


# ---------------------------------------------------------------------------
# Upsampler: PixelShuffle-based 2x
# ---------------------------------------------------------------------------
class PixelShuffleUpsampler(nn.Module):
    def __init__(self, channels, scale=2):
        super().__init__()
        # For 2x, one PixelShuffle block.
        # For 4x, stack two (not needed here).
        layers = []
        num_shuffles = int(math.log2(scale))
        for _ in range(num_shuffles):
            layers += [
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Full TSRN
# ---------------------------------------------------------------------------
class TSRN(nn.Module):
    """Text Super-Resolution Network.

    Args:
        in_channels: 3 for RGB only, 4 for RGB + mask.
        out_channels: typically 3 (RGB output).
        hidden_channels: internal feature dim, default 64.
        num_blocks: how many RecurrentResidualBlocks, default 5.
        scale: upsampling factor, default 2 (16->32 height, 64->128 width).
        use_stn: if True, apply an STN rectification at the front.
        prior_channels: if > 0, expects an additional feature map of this
            many channels to be concatenated into the features partway
            through. Used by the TPGSR wrapper to inject text priors.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        hidden_channels=64,
        num_blocks=5,
        scale=2,
        use_stn=True,
        prior_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.scale = scale
        self.prior_channels = prior_channels

        self.stn = STN(in_channels) if use_stn else None

        # Front: conv from in_channels -> hidden_channels
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # Residual body
        self.blocks = nn.ModuleList([
            RecurrentResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # After the body, an optional prior-fusion layer.
        # If prior_channels > 0, the TPGSR wrapper injects a (B, P, H, W) feature map
        # here; this conv mixes it back into hidden_channels features.
        if prior_channels > 0:
            self.prior_fusion = nn.Sequential(
                nn.Conv2d(hidden_channels + prior_channels, hidden_channels, kernel_size=3, padding=1),
                nn.PReLU(),
            )
        else:
            self.prior_fusion = None

        # Tail: merge residual, upsample, project to out_channels
        self.tail_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
        )
        self.upsampler = PixelShuffleUpsampler(hidden_channels, scale=scale)
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=9, padding=4)

    def forward(self, x, prior_map=None):
        """
        Args:
            x: (B, in_channels, 16, 64)
            prior_map: optional (B, prior_channels, 16, 64) text-prior feature map.

        Returns:
            SR output in (-1, 1) range via tanh: (B, out_channels, 32, 128)
        """
        if self.stn is not None:
            x = self.stn(x)

        feat = self.head(x)               # (B, C, 16, 64)
        residual_head = feat

        for block in self.blocks:
            feat = block(feat)

        if self.prior_fusion is not None:
            if prior_map is None:
                # Zero-filled prior when the wrapper hasn't computed one yet
                # (e.g., iteration 1 of TPGSR).
                b, _, h, w = feat.shape
                prior_map = feat.new_zeros(b, self.prior_channels, h, w)
            feat = torch.cat([feat, prior_map], dim=1)
            feat = self.prior_fusion(feat)

        feat = self.tail_conv(feat)
        feat = feat + residual_head        # long skip

        up = self.upsampler(feat)          # (B, C, 32, 128)
        out = self.output_conv(up)         # (B, out_channels, 32, 128)
        return torch.tanh(out)