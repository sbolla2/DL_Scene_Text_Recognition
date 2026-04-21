"""
Phase 2 — SR cache configuration for TRBA variant.

Runs the trained TRBA-guided TSRN on all crops in the heavy benchmark,
saves SR outputs to disk.

Hybrid inference: crops outside [MIN, MAX] width range skip TSRN (passthrough).
"""

import torch


class SRCacheTRBAConfig:
    # ---- Checkpoints ----
    TSRN_CHECKPOINT_PATH = '/content/weights/tsrn_heavy_trba_best.pth'
    TPG_CHECKPOINT_PATH = '/content/weights/trba_tpg_clean_best.pth'

    # ---- SR output location ----
    SR_CACHE_ROOT = '/content/data/sr_cache_trba/heavy'

    # ---- Dimensions (same as Phase 1) ----
    LR_HEIGHT = 16
    LR_WIDTH = 64
    HR_HEIGHT = 32
    HR_WIDTH = 128
    IN_CHANNELS = 4
    OUT_CHANNELS = 3

    # Architecture
    TSRN_HIDDEN_CHANNELS = 64
    TSRN_NUM_BLOCKS = 5
    TSRN_STN = True
    NUM_ITERATIONS = 2
    PRIOR_FEAT_CHANNELS = 32

    # TPG config
    TPG_NUM_CLASSES = 38
    TPG_SEQ_LEN = 26
    BATCH_MAX_LENGTH = 25

    # TRBA architecture mirrors (for load_trba_tpg)
    TRBA_IMG_HEIGHT = 32
    TRBA_IMG_WIDTH = 100
    TRBA_IMG_CHANNELS = 1
    TRBA_NUM_FIDUCIAL = 20
    TRBA_CNN_OUT_CHANNELS = 512
    TRBA_RNN_HIDDEN_SIZE = 256

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

    # ---- Hybrid passthrough range ----
    MIN_CROP_WIDTH = 40
    MAX_CROP_WIDTH = 160

    # ---- Inference ----
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    SAMPLE_DUMP_COUNT = 20

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
