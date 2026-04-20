"""
Phase 3 — Fine-tune heavy-trained TRBA on SR outputs.
"""

import torch


class FinetuneTRBAConfig:
    BASELINE_CHECKPOINT_PATH = '/content/weights/trba_totaltext_heavy_best.pth'
    SR_CACHE_ROOT = '/content/data/sr_cache_trba/heavy'

    NUM_EPOCHS = 6
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    LR = 2e-4
    LR_MIN = 1e-6
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 5.0

    # TRBA architecture
    IMG_HEIGHT = 32
    IMG_WIDTH = 100
    IMG_CHANNELS = 1
    NUM_FIDUCIAL = 20
    BATCH_MAX_LENGTH = 25
    CNN_OUT_CHANNELS = 512
    RNN_HIDDEN_SIZE = 256

    SAVE_PATH = '/content/weights/trba_tpgsr_finetuned_best.pth'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
