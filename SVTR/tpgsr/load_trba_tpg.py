"""
Load frozen TRBA-TPG for use in Phase 1 (training) and Phase 2 (SR caching).

Instantiates a ClovaaiTRBA, loads the Phase 0 checkpoint, puts it in eval mode,
and freezes all parameters.
"""

import torch

from __main__ import ClovaaiTRBA, trba_converter


def load_frozen_trba_tpg(checkpoint_path, cfg, device=None):
    """
    Args:
        checkpoint_path: path to trba_tpg_clean_best.pth
        cfg: any config with TRBA architecture fields (IMG_WIDTH, IMG_CHANNELS,
             NUM_FIDUCIAL, BATCH_MAX_LENGTH, IMG_HEIGHT, CNN_OUT_CHANNELS,
             RNN_HIDDEN_SIZE, DEVICE)
        device: override device; defaults to cfg.DEVICE

    Returns:
        Frozen TRBA model in eval mode on the specified device.
    """
    device = device or cfg.DEVICE

    tpg = ClovaaiTRBA(num_class=len(trba_converter.character), cfg=cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tpg.load_state_dict(ckpt['model_state_dict'])
    tpg.eval()

    for p in tpg.parameters():
        p.requires_grad = False

    print(
        f"TRBA-TPG loaded from {checkpoint_path}\n"
        f"  checkpoint epoch: {ckpt['epoch']}, val_acc: {ckpt['val_acc']:.1f}%, "
        f"val_cer: {ckpt['val_cer']:.1f}%"
    )
    return tpg
