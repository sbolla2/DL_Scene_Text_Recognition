"""
Phase 0 — TRBA-TPG configuration.

Trains a TRBA model on CLEAN crops to serve as the frozen Text Prior Generator
for TRBA-TPGSR. Analogous to the CRNN-TPG v2 setup but with TRBA's architecture
and TRBA-specific schedules.

Key differences from baseline TRBAConfig:
  1. Longer training (8 epochs vs 5) to match clean-adaptation rhythm
  2. Mild augmentation enabled (reduces train/val gap)
  3. Save path separates the clean TPG from the heavy baseline
"""

from __main__ import Config


class TRBATPGConfig(Config):
    # TRBA-specific dims (match your TRBAConfig)
    IMG_WIDTH = 100
    IMG_CHANNELS = 1
    NUM_FIDUCIAL = 20
    BATCH_MAX_LENGTH = 25
    CHARACTER = '0123456789abcdefghijklmnopqrstuvwxyz'

    # Schedule
    NUM_EPOCHS = 8
    BATCH_SIZE = 64
    LR = 2e-4
    LR_STEP = 3
    LR_GAMMA = 0.5
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    # Mild augmentation — same philosophy as CRNN-TPG v2
    AUGMENTATION = {
        'affine_degrees': 2.0,
        'affine_shear': 4.0,
        'brightness': 0.15,
        'contrast': 0.15,
        'color_prob': 0.30,
        'blur_kernel': 3,
        'blur_sigma': (0.1, 0.3),
        'blur_prob': 0.10,
    }

    SAVE_PATH = '/content/weights/trba_tpg_clean_best.pth'
