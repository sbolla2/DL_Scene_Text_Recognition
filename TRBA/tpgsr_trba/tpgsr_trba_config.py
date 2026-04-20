"""
Phase 1 — TPGSR configuration for TRBA.

Mirrors TPGSRConfig but with:
  - CE loss instead of CTC for the text component
  - TRBA input dims (32x100 grayscale) for the TPG adapter
  - TRBA charset size (38 classes = 36 + [GO] + [s])
  - TRBA sequence length (26 = BATCH_MAX_LENGTH + 1)
"""

import torch


class TPGSRTRBAConfig:
    # ---- SR dimensions (same as CRNN-TPGSR) ----
    LR_HEIGHT = 16
    LR_WIDTH = 64
    HR_HEIGHT = 32
    HR_WIDTH = 128
    IN_CHANNELS = 4
    OUT_CHANNELS = 3
    SR_SCALE = 2

    # TSRN backbone
    TSRN_HIDDEN_CHANNELS = 64
    TSRN_NUM_BLOCKS = 5
    TSRN_STN = True

    # ---- TPGSR refinement ----
    NUM_ITERATIONS = 2

    # ---- Text prior (TRBA) ----
    TPG_NUM_CLASSES = 38       # 36 chars + [GO] + [s]
    TPG_SEQ_LEN = 26           # BATCH_MAX_LENGTH (25) + 1
    TPG_CHECKPOINT_PATH = '/content/weights/trba_tpg_clean_best.pth'

    # TRBA input dimensions
    TRBA_IMG_HEIGHT = 32
    TRBA_IMG_WIDTH = 100
    TRBA_IMG_CHANNELS = 1
    TRBA_NUM_FIDUCIAL = 20
    TRBA_BATCH_MAX_LENGTH = 25
    TRBA_CNN_OUT_CHANNELS = 512   # ResNet output
    TRBA_RNN_HIDDEN_SIZE = 256    # BiLSTM hidden

    # Used by AttnLabelConverter — [GO] is at index 0
    GO_IDX = 0

    # ---- Width filter for training ----
    MIN_TRAIN_WIDTH = 40
    MAX_TRAIN_WIDTH = 160

    # ---- Loss weights ----
    L1_WEIGHT = 1.0
    MSE_WEIGHT = 1.0
    CE_WEIGHT = 0.1           # analog to CRNN's CTC weight

    # ---- Optimizer ----
    LR = 1e-4
    LR_MIN = 1e-6
    BETAS = (0.9, 0.999)
    WEIGHT_DECAY = 0.0
    GRAD_CLIP = 1.0

    # ---- Training schedule ----
    NUM_EPOCHS = 30
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # ---- Text encoding ----
    BATCH_MAX_LENGTH = 25

    # ---- Checkpointing / logging ----
    SAVE_PATH = '/content/weights/tsrn_heavy_trba_best.pth'
    LOG_INTERVAL_BATCHES = 50

    # ---- Runtime ----
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42

    # ---- IMG_HEIGHT alias for ClovaaiTRBA ctor compatibility ----
    # (ClovaaiTRBA reads cfg.IMG_HEIGHT / cfg.IMG_WIDTH etc.)
    @property
    def IMG_HEIGHT(self):
        return self.TRBA_IMG_HEIGHT

    @property
    def IMG_WIDTH(self):
        return self.TRBA_IMG_WIDTH

    @property
    def IMG_CHANNELS(self):
        return self.TRBA_IMG_CHANNELS

    @property
    def NUM_FIDUCIAL(self):
        return self.TRBA_NUM_FIDUCIAL

    @property
    def CNN_OUT_CHANNELS(self):
        return self.TRBA_CNN_OUT_CHANNELS

    @property
    def RNN_HIDDEN_SIZE(self):
        return self.TRBA_RNN_HIDDEN_SIZE
