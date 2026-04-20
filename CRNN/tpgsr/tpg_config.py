"""
Phase 0 v2 — TPG-CRNN configuration with regularization.

Trains a CRNN on CLEAN crops to serve as the frozen Text Prior Generator.

Changes from v1:
  1. Mild augmentation enabled (was None) — reduces overfitting
  2. LR_STEP = 4 instead of 8 — earlier LR decay
  3. WEIGHT_DECAY = 1e-4 instead of 5e-5 — stronger L2 regularization
  4. SAVE_PATH = crnn_tpg_clean_v2_best.pth — keeps v1 for comparison
"""

from __main__ import Config


class TPGCRNNConfig(Config):
    # Schedule
    NUM_EPOCHS = 12
    LR = 2e-4
    LR_STEP = 4             # v1: 8 — decay every 4 epochs instead of 8
    LR_GAMMA = 0.5

    # Regularization
    WEIGHT_DECAY = 1e-4     # v1: 5e-5 — stronger L2
    FREEZE_CNN_EPOCHS = 1

    # Mild augmentation — enabled in v2 to reduce the ~17-point train/val gap from v1
    # These values are deliberately small: we want the TPG to stay calibrated to the
    # clean distribution, just with a little robustness to tiny variations.
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

    # Separate v2 checkpoint — keeps v1 intact for comparison
    SAVE_PATH = '/content/weights/crnn_tpg_clean_v2_best.pth'
