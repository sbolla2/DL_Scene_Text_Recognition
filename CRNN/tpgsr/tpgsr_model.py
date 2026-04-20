"""
TPGSR model wrapper.

Combines:
  - TSRN (super-resolution backbone, trainable)
  - Frozen CRNN-TPG (text prior generator, read-only)
  - Iterative refinement loop (paper default: 2 iterations)

The core idea:
  Iter 1: TSRN(LR, prior=None) -> SR_1
          TPG(SR_1) -> prior_1 (shape (T, B, num_classes))
  Iter 2: TSRN(LR, prior=prior_1_map) -> SR_2
          TPG(SR_2) -> prior_2
  Return: SR_2 (final), plus all priors for loss computation.

The prior is a character probability sequence (T, B, num_classes). To inject
it into TSRN's spatial features, we first project it to a small feature vector
and tile it spatially to match TSRN's internal feature map.

Gradients:
  - TSRN: trainable, gradients flow normally.
  - TPG: frozen (requires_grad=False), but gradients do flow THROUGH it back
    into TSRN. This is the key — TSRN learns to produce images that the TPG
    can read well, even though the TPG itself is fixed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorSpatialProjector(nn.Module):
    """Convert (T, B, num_classes) prior into (B, prior_feat_channels, H, W) map.

    Process:
      1. Average-pool the sequence (T, B, C) -> (B, C) to get a per-sample vector.
         This is a simple summary; the paper uses a slightly more involved fusion
         but average-pool is a good baseline.
      2. Linear project to prior_feat_channels.
      3. Unsqueeze and tile to (B, prior_feat_channels, H, W).
    """

    def __init__(self, num_classes, prior_feat_channels, spatial_size):
        super().__init__()
        self.prior_feat_channels = prior_feat_channels
        self.h, self.w = spatial_size
        self.proj = nn.Sequential(
            nn.Linear(num_classes, prior_feat_channels),
            nn.PReLU(),
            nn.Linear(prior_feat_channels, prior_feat_channels),
        )

    def forward(self, prior):
        """
        Args:
            prior: (T, B, num_classes) log-probabilities from CRNN TPG.

        Returns:
            (B, prior_feat_channels, H, W) spatial feature map.
        """
        # (T, B, C) -> (B, C) via mean over time
        pooled = prior.mean(dim=0)                           # (B, C)
        # Exp the log-probs so we work in probability space (mean pooled log probs
        # can be very negative; probabilities are easier to learn from).
        probs = pooled.exp()
        feat = self.proj(probs)                              # (B, prior_feat_channels)
        b, c = feat.shape
        # Tile spatially
        spatial = feat.view(b, c, 1, 1).expand(b, c, self.h, self.w)
        return spatial


class TPGAdapter(nn.Module):
    """Adapts a CRNN to consume TSRN's 3-channel RGB output.

    The CRNN was trained on 1-channel grayscale inputs at 32x256. TSRN
    produces 3-channel RGB at 32x128. We:
      1. Convert RGB to grayscale (simple luma).
      2. Resize 32x128 -> 32x256 (width stretch, matches CRNN's expected width).
      3. Renormalize from TSRN's tanh range [-1, 1] to CRNN's [-1, 1]
         (same, no-op — but explicit for clarity).

    The CRNN is frozen (set at construction by the caller).
    """

    def __init__(self, crnn_model, target_height=32, target_width=256):
        super().__init__()
        self.crnn = crnn_model  # already frozen + eval mode by the caller
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, sr_rgb):
        """
        Args:
            sr_rgb: (B, 3, 32, 128) in tanh range [-1, 1].

        Returns:
            log_probs: (T, B, num_classes) from CRNN.
        """
        # RGB -> grayscale (ITU-R 601 luma weights).
        # sr_rgb is in [-1, 1]; shift to [0, 1] first for correct luma, then back.
        rgb_01 = (sr_rgb + 1.0) * 0.5
        r, g, b = rgb_01[:, 0:1], rgb_01[:, 1:2], rgb_01[:, 2:3]
        gray_01 = 0.299 * r + 0.587 * g + 0.114 * b           # (B, 1, 32, 128)
        gray = gray_01 * 2.0 - 1.0                            # back to [-1, 1]

        # Resize width 128 -> 256 to match CRNN's training width.
        gray_resized = F.interpolate(
            gray,
            size=(self.target_height, self.target_width),
            mode='bicubic',
            align_corners=False,
        )

        # CRNN forward: (B, 1, 32, 256) -> (T, B, num_classes) log-softmax
        return self.crnn(gray_resized)


class TPGSR(nn.Module):
    """End-to-end TPGSR model.

    Args:
        tsrn: instantiated TSRN with prior_channels > 0.
        tpg: frozen CRNN TPG (requires_grad=False, eval mode).
        num_classes: TPG output dim (37 for your 36 chars + blank).
        num_iterations: refinement iterations (2 is the paper default).
        prior_feat_channels: how many channels the prior map has after projection.
        lr_spatial_size: (H, W) of TSRN's internal feature map where prior is fused.
            For 16x64 input that's (16, 64) in our implementation.
    """

    def __init__(
        self,
        tsrn,
        tpg,
        num_classes=37,
        num_iterations=2,
        prior_feat_channels=32,
        lr_spatial_size=(16, 64),
    ):
        super().__init__()
        self.tsrn = tsrn
        self.num_iterations = num_iterations
        self.num_classes = num_classes

        self.prior_proj = PriorSpatialProjector(
            num_classes=num_classes,
            prior_feat_channels=prior_feat_channels,
            spatial_size=lr_spatial_size,
        )
        self.tpg_adapter = TPGAdapter(tpg)

        # Sanity: TSRN must have been built with prior_channels == prior_feat_channels
        assert tsrn.prior_channels == prior_feat_channels, (
            f'TSRN.prior_channels ({tsrn.prior_channels}) must equal '
            f'prior_feat_channels ({prior_feat_channels})'
        )

    def forward(self, lr):
        """
        Args:
            lr: (B, 4, 16, 64) degraded input + polygon mask.

        Returns:
            dict with:
              - 'sr_iterations': list of SR outputs per iter, each (B, 3, 32, 128)
              - 'prior_iterations': list of TPG log-probs per iter, each (T, B, num_classes)
              - 'final_sr': the last iteration's SR output
        """
        sr_outputs = []
        prior_outputs = []

        prior_map = None  # first iteration has no prior

        for iteration in range(self.num_iterations):
            sr = self.tsrn(lr, prior_map=prior_map)           # (B, 3, 32, 128)
            sr_outputs.append(sr)

            # Run TPG on this SR output to get priors for the next iteration.
            # (Also included for the final iter because we want it for loss.)
            log_probs = self.tpg_adapter(sr)                  # (T, B, num_classes)
            prior_outputs.append(log_probs)

            # Project this iteration's prior into a feature map for the next pass.
            if iteration < self.num_iterations - 1:
                prior_map = self.prior_proj(log_probs)        # (B, P, 16, 64)

        return {
            'sr_iterations': sr_outputs,
            'prior_iterations': prior_outputs,
            'final_sr': sr_outputs[-1],
        }
