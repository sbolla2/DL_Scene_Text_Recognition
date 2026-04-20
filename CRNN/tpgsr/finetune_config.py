"""
Phase 3 — CRNN fine-tuning on SR outputs.

Takes the heavy-trained CRNN baseline and continues training on the SR cache
(Phase 2 output). This adapts the recognizer from raw-degraded to SR-output
distribution — the key step that addresses the train/test mismatch seen in
naive TPGSR deployments.

Hybrid input policy (matches Phase 2):
  - If record has sr_applied=True: use SR output PNG
  - Else (passthrough): use the original degraded crop
"""

import torch


class FinetuneConfig:
    # ---- Starting checkpoint (heavy-trained CRNN baseline) ----
    BASELINE_CHECKPOINT_PATH = '/content/weights/crnn_totaltext_heavy_best.pth'

    # ---- SR cache location (Phase 2 output) ----
    SR_CACHE_ROOT = '/content/data/sr_cache/heavy'

    # ---- Training schedule ----
    NUM_EPOCHS = 6              # small number — just adapt, don't overfit
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    LR = 2e-4                    # lower than baseline training (fine-tuning)
    LR_MIN = 1e-6
    WEIGHT_DECAY = 1e-4

    # Freeze VGG CNN for first N epochs (same idea as Phase 0)
    FREEZE_CNN_EPOCHS = 1

    # ---- Output checkpoint ----
    SAVE_PATH = '/content/drive/MyDrive/tpgsr_project/crnn_tpgsr_finetuned_best.pth'

    # ---- Runtime ----
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
