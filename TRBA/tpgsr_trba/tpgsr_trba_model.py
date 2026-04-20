"""
TPGSR wrapper for TRBA.

Combines:
  - TSRN (super-resolution backbone, trainable, same as CRNN variant)
  - Frozen TRBA (text prior generator, read-only, 38-class attention output)
  - Iterative refinement loop (2 iterations)

Differences vs. CRNN TPGSR wrapper:
  - TPGAdapter resizes SR output to 32x100 (not 32x256) for TRBA
  - TPG output is (B, T=26, 38) raw logits, not (T=32, B, 37) log-softmax
  - PriorSpatialProjector handles variable-length attention output:
    weights steps by their max softmax probability to down-weight padding steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPriorProjector(nn.Module):
    """Convert TRBA's (B, T, num_classes) attention logits into (B, P, H, W).

    Strategy:
      1. Softmax across classes to get per-step prob distributions.
      2. Max-probability-weighted pool across T dim.
         (Steps with confident predictions contribute more; padding-like
         steps contribute less.)
      3. Linear project to prior_feat_channels.
      4. Tile spatially to (B, P, H, W).
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

    def forward(self, logits):
        """
        Args:
            logits: (B, T, num_classes) raw attention logits from TRBA.

        Returns:
            (B, prior_feat_channels, H, W) spatial feature map.
        """
        probs = F.softmax(logits, dim=2)                   # (B, T, C)
        # Per-step confidence = max probability
        step_conf = probs.max(dim=2).values                # (B, T)
        # Normalize weights along T so they sum to 1 per sample
        weights = step_conf / (step_conf.sum(dim=1, keepdim=True) + 1e-6)
        weights = weights.unsqueeze(-1)                     # (B, T, 1)
        # Weighted mean across T
        pooled = (probs * weights).sum(dim=1)              # (B, C)

        feat = self.proj(pooled)                           # (B, P)
        b, c = feat.shape
        return feat.view(b, c, 1, 1).expand(b, c, self.h, self.w)


class TRBATPGAdapter(nn.Module):
    """Adapts TSRN's 3-ch RGB SR output to TRBA's 1-ch 32x100 grayscale input.

    Process:
      1. RGB -> grayscale via luma weights
      2. Resize 32x128 -> 32x100
      3. Convert tanh [-1, 1] to TRBA's expected range (also [-1, 1] with
         same normalization as AttnDataset uses)

    TRBA forward requires a 'text' argument even in inference mode — we pass
    a zero placeholder (the autoregressive decoder generates its own input).
    """

    def __init__(self, trba_model, target_height=32, target_width=100,
                 batch_max_length=25):
        super().__init__()
        self.trba = trba_model  # already frozen + eval by caller
        self.target_height = target_height
        self.target_width = target_width
        self.batch_max_length = batch_max_length

    def forward(self, sr_rgb):
        """
        Args:
            sr_rgb: (B, 3, 32, 128) in tanh range [-1, 1].

        Returns:
            logits: (B, T=26, num_classes=38) raw TRBA attention outputs.
        """
        # RGB -> grayscale
        rgb_01 = (sr_rgb + 1.0) * 0.5
        r, g, b = rgb_01[:, 0:1], rgb_01[:, 1:2], rgb_01[:, 2:3]
        gray_01 = 0.299 * r + 0.587 * g + 0.114 * b
        gray = gray_01 * 2.0 - 1.0                          # (B, 1, 32, 128)

        # Resize width to TRBA's 100
        gray_resized = F.interpolate(
            gray,
            size=(self.target_height, self.target_width),
            mode='bicubic',
            align_corners=False,
        )

        # Run TRBA in inference mode — no text needed, decoder uses its own predictions
        b = gray_resized.size(0)
        text_placeholder = torch.zeros(
            b, self.batch_max_length + 1,
            dtype=torch.long, device=gray_resized.device,
        )
        return self.trba(gray_resized, text_placeholder, is_train=False)


class TPGSRTRBA(nn.Module):
    """End-to-end TPGSR model with TRBA as the frozen TPG.

    Args:
        tsrn: instantiated TSRN with prior_channels > 0
        trba: frozen TRBA TPG
        num_classes: 38 (36 + [GO] + [s])
        num_iterations: 2 (paper default)
        prior_feat_channels: 32 (matches CRNN variant)
        lr_spatial_size: (H, W) of TSRN's feature map where prior is fused
        trba_image_size: (H, W) TRBA expects (32, 100)
        batch_max_length: 25 (TRBA's max decoded length)
    """

    def __init__(
        self,
        tsrn,
        trba,
        num_classes=38,
        num_iterations=2,
        prior_feat_channels=32,
        lr_spatial_size=(16, 64),
        trba_image_size=(32, 100),
        batch_max_length=25,
    ):
        super().__init__()
        self.tsrn = tsrn
        self.num_iterations = num_iterations
        self.num_classes = num_classes

        self.prior_proj = AttentionPriorProjector(
            num_classes=num_classes,
            prior_feat_channels=prior_feat_channels,
            spatial_size=lr_spatial_size,
        )
        self.tpg_adapter = TRBATPGAdapter(
            trba,
            target_height=trba_image_size[0],
            target_width=trba_image_size[1],
            batch_max_length=batch_max_length,
        )

        assert tsrn.prior_channels == prior_feat_channels, (
            f'TSRN.prior_channels ({tsrn.prior_channels}) must equal '
            f'prior_feat_channels ({prior_feat_channels})'
        )

    def forward(self, lr, text_gt=None):
        """
        Args:
            lr: (B, 4, 16, 64) — LR + polygon mask
            text_gt: (B, BATCH_MAX_LENGTH+2) ground-truth TRBA-encoded text,
                     used only for teacher-forced loss computation during
                     training. If provided, additionally returns TRBA's
                     teacher-forced logits for the CE loss.

        Returns:
            dict with:
              - 'sr_iterations': list of SR outputs per iter (B, 3, 32, 128)
              - 'prior_iterations': list of inference-mode TRBA logits per iter
                                    (B, 26, 38) — used for prior projection
              - 'teacher_forced_iterations': if text_gt given, list of teacher-
                                             forced TRBA logits per iter for CE loss
              - 'final_sr': last iteration's SR
        """
        sr_outputs = []
        prior_outputs = []
        teacher_forced_outputs = []

        prior_map = None

        for iteration in range(self.num_iterations):
            sr = self.tsrn(lr, prior_map=prior_map)         # (B, 3, 32, 128)
            sr_outputs.append(sr)

            # Inference-mode TRBA for prior generation
            inference_logits = self.tpg_adapter(sr)          # (B, 26, 38)
            prior_outputs.append(inference_logits)

            # Teacher-forced TRBA for loss computation (only if GT text provided)
            if text_gt is not None:
                # Adapt SR to TRBA input
                rgb_01 = (sr + 1.0) * 0.5
                r, g, b = rgb_01[:, 0:1], rgb_01[:, 1:2], rgb_01[:, 2:3]
                gray_01 = 0.299 * r + 0.587 * g + 0.114 * b
                gray = gray_01 * 2.0 - 1.0
                gray_resized = F.interpolate(
                    gray,
                    size=(self.tpg_adapter.target_height,
                          self.tpg_adapter.target_width),
                    mode='bicubic',
                    align_corners=False,
                )
                tf_logits = self.tpg_adapter.trba(
                    gray_resized,
                    text_gt[:, :-1],   # teacher forcing input
                    is_train=True,
                )
                teacher_forced_outputs.append(tf_logits)

            # Project inference-mode prior to spatial map for next TSRN pass
            if iteration < self.num_iterations - 1:
                prior_map = self.prior_proj(inference_logits)

        out = {
            'sr_iterations': sr_outputs,
            'prior_iterations': prior_outputs,
            'final_sr': sr_outputs[-1],
        }
        if text_gt is not None:
            out['teacher_forced_iterations'] = teacher_forced_outputs
        return out
