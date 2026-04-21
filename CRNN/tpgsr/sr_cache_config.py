"""
Phase 2 — SR cache configuration.

Runs trained TSRN on all crops in the heavy benchmark, saves SR outputs to disk,
and creates per-split manifests that Phase 3 fine-tuning will read.

Hybrid inference policy:
  - Crops with original clean-crop width in [MIN, MAX]: run through TSRN
  - Crops outside that range: passthrough (no SR applied)

This mirrors the width filter used during Phase 1 training. Out-of-range crops
weren't seen during TSRN training, so TSRN would produce unreliable outputs.
At inference, we fall back to the raw degraded input for those.
"""

import os
import torch


class SRCacheConfig:
    # ---- TSRN checkpoint ----
    TSRN_CHECKPOINT_PATH = '/content/weights/tsrn_heavy_best.pth'
    TPG_CHECKPOINT_PATH = '/content/weights/crnn_tpg_clean_v2_best.pth'

    # ---- SR output location ----
    SR_CACHE_ROOT = '/content/data/sr_cache/heavy'

    # ---- Same dimensions as Phase 1 ----
    LR_HEIGHT = 16
    LR_WIDTH = 64
    HR_HEIGHT = 32
    HR_WIDTH = 128
    IN_CHANNELS = 4
    OUT_CHANNELS = 3

    # ---- Architecture config (must match Phase 1's TSRN) ----
    TSRN_HIDDEN_CHANNELS = 64
    TSRN_NUM_BLOCKS = 5
    TSRN_STN = True
    NUM_ITERATIONS = 2
    PRIOR_FEAT_CHANNELS = 32

    # ---- TPG config ----
    TPG_NUM_CLASSES = 37
    TPG_SEQ_LEN = 32

    # ---- Hybrid passthrough range (same as Phase 1 training width filter) ----
    MIN_CROP_WIDTH = 40
    MAX_CROP_WIDTH = 160

    # ---- Inference batch size (larger OK since no grads) ----
    BATCH_SIZE = 128
    NUM_WORKERS = 4

    # ---- Visual sample dump ----
    SAMPLE_DUMP_COUNT = 20     # saves 20 (degraded|SR|clean) triplets as a grid

    # ---- Runtime ----
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
