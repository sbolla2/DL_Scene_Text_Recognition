"""
Phase 0 — TPG-CRNN training driver.

Trains a CRNN on clean crops to serve as the frozen Text Prior Generator
inside TPGSR (Phase 1).

Expects the baseline notebook to have run first in the same Colab session,
producing final_{train,val,test}_records and all CRNN definitions in the
__main__ namespace.

Usage:
    import sys
    sys.path.insert(0, '/content')
    from tpgsr.train_tpg import train_tpg_crnn
    from tpgsr.tpg_config import TPGCRNNConfig

    tpg_cfg = TPGCRNNConfig()
    tpg_cfg.SAVE_PATH = '/content/drive/MyDrive/tpgsr_project/crnn_tpg_clean_v2_best.pth'
    tpg_model, tpg_test_metrics = train_tpg_crnn(tpg_cfg)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from __main__ import (
    final_train_records,
    final_val_records,
    final_test_records,
    prepare_crnn_records,
    label_encoder,
    build_pretrained_crnn,
    train_with_pretrained,
    evaluate,
    collate_fn,
)

from tpgsr.tpg_config import TPGCRNNConfig
from tpgsr.datasets import CleanCropDataset


def build_clean_loaders(tpg_cfg):
    """Normalize records (charset filter + length filter) and wrap in loaders.

    Train split uses augmentation IFF cfg.AUGMENTATION is not None.
    Val/test always use augment=False (clean distribution eval).
    """
    clean_train = prepare_crnn_records(final_train_records, 'TPG-CRNN clean train')
    clean_val = prepare_crnn_records(final_val_records, 'TPG-CRNN clean val')
    clean_test = prepare_crnn_records(final_test_records, 'TPG-CRNN clean test')

    train_augment = tpg_cfg.AUGMENTATION is not None
    print(f'Train augmentation: {"ON" if train_augment else "OFF"}')

    train_ds = CleanCropDataset(clean_train, label_encoder, tpg_cfg, augment=train_augment)
    val_ds = CleanCropDataset(clean_val, label_encoder, tpg_cfg, augment=False)
    test_ds = CleanCropDataset(clean_test, label_encoder, tpg_cfg, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=tpg_cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=tpg_cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tpg_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=tpg_cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tpg_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=tpg_cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f'TPG train batches: {len(train_loader)}')
    print(f'TPG val batches:   {len(val_loader)}')
    print(f'TPG test batches:  {len(test_loader)}')
    return train_loader, val_loader, test_loader


def train_tpg_crnn(tpg_cfg=None):
    """Run Phase 0 end to end.

    Returns the trained model (also saved to tpg_cfg.SAVE_PATH) and the test metrics.
    """
    tpg_cfg = tpg_cfg or TPGCRNNConfig()
    print('=' * 70)
    print('Phase 0 — Training TPG-CRNN on CLEAN crops')
    print('=' * 70)
    print(f'Epochs:       {tpg_cfg.NUM_EPOCHS}')
    print(f'Freeze CNN:   {tpg_cfg.FREEZE_CNN_EPOCHS} epoch(s)')
    print(f'Batch size:   {tpg_cfg.BATCH_SIZE}')
    print(f'Base LR:      {tpg_cfg.LR}')
    print(f'LR step:      {tpg_cfg.LR_STEP} (gamma={tpg_cfg.LR_GAMMA})')
    print(f'Weight decay: {tpg_cfg.WEIGHT_DECAY}')
    print(f'Augment:      {"ON" if tpg_cfg.AUGMENTATION is not None else "OFF"}')
    print(f'Save path:    {tpg_cfg.SAVE_PATH}')
    print(f'Device:       {tpg_cfg.DEVICE}')
    print('-' * 70)

    train_loader, val_loader, test_loader = build_clean_loaders(tpg_cfg)

    model = build_pretrained_crnn(label_encoder, tpg_cfg)
    train_with_pretrained(model, train_loader, val_loader, tpg_cfg, label_encoder)

    # Reload best checkpoint (train_with_pretrained leaves model in final-epoch state)
    ckpt = torch.load(tpg_cfg.SAVE_PATH, map_location=tpg_cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(
        f"\nLoaded best TPG checkpoint: epoch {ckpt['epoch']}, "
        f"val_acc={ckpt['val_acc']:.1f}%, val_cer={ckpt['val_cer']:.1f}%"
    )

    criterion = nn.CTCLoss(blank=tpg_cfg.BLANK_IDX, reduction='mean', zero_infinity=True)
    test_metrics = evaluate(model, test_loader, criterion, tpg_cfg.DEVICE, label_encoder)
    print('=' * 70)
    print(
        f"TPG-CRNN on CLEAN test -> "
        f"Word Acc: {test_metrics['word_acc']:.1f}%, "
        f"CER: {test_metrics['cer']:.1f}%, "
        f"Loss: {test_metrics['loss']:.4f}"
    )
    print('=' * 70)
    print('Phase 0 complete. This checkpoint will be the frozen TPG in Phase 1.')

    return model, test_metrics
