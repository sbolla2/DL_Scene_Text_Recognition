"""
Phase 1 — TPGSR configuration.

All hyperparameters for training TSRN with frozen CRNN-TPG.

Key values:
  - LR 16x64 -> HR 32x128 (paper-standard)
  - 4-channel input (RGB + polygon mask)
  - 2 iterations of refinement
  - Loss weights: L1=1.0, MSE=1.0, CTC=0.1
  - 30 epochs
"""

import torch


class TPGSRConfig:
    # ---- Architecture ----
    LR_HEIGHT = 16
    LR_WIDTH = 64
    HR_HEIGHT = 32
    HR_WIDTH = 128
    IN_CHANNELS = 4          # RGB + binary polygon mask
    OUT_CHANNELS = 3         # RGB output
    SR_SCALE = 2             # 16x64 -> 32x128

    # TSRN backbone
    TSRN_HIDDEN_CHANNELS = 64
    TSRN_NUM_BLOCKS = 5
    TSRN_STN = True          # use spatial transformer for alignment (paper default)

    # ---- TPGSR refinement ----
    NUM_ITERATIONS = 2       # paper default

    # ---- Text prior ----
    # Charset size must match the TPG's output dim.
    TPG_NUM_CLASSES = 37     # 36 chars + 1 blank (CTC)
    # Sequence length the TPG emits on 32x128 input — CRNN time steps = W/4 = 32.
    TPG_SEQ_LEN = 32
    TPG_CHECKPOINT_PATH = '/content/weights/crnn_tpg_clean_v2_best.pth'

    # ---- Width filter for training ----
    # Only train TSRN on crops whose original (pre-resize) width is in this range.
    # Crops outside this range will bypass TSRN at inference time (Phase 2).
    MIN_TRAIN_WIDTH = 40
    MAX_TRAIN_WIDTH = 160

    # ---- Loss weights ----
    L1_WEIGHT = 1.0
    MSE_WEIGHT = 1.0
    # CTC_WEIGHT = 0.1
    CTC_WEIGHT = 0.1

    # ---- Optimizer ----
    LR = 1e-4
    LR_MIN = 1e-6            # cosine anneal floor
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 0.0       # TSRN is already regularized by its skip connections
    GRAD_CLIP = 1.0

    # ---- Training schedule ----
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # ---- Text encoding ----
    BATCH_MAX_LENGTH = 25
    BLANK_IDX = 0

    # ---- Checkpointing / logging ----
    SAVE_PATH = '/content/weights/tsrn_heavy_best.pth'
    LOG_INTERVAL_BATCHES = 50    # print a mid-epoch progress line every N batches
    SAMPLE_DIR = None            # set to a path to dump visual samples each epoch

    # ---- Runtime ----
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
