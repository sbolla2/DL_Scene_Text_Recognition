"""
Phase 0 — TRBA-TPG training driver.

Trains TRBA on clean crops to serve as the frozen Text Prior Generator
inside TRBA-TPGSR (Phase 1).

Usage:
    from tpgsr.train_trba_tpg import train_trba_tpg
    from tpgsr.trba_tpg_config import TRBATPGConfig
    cfg = TRBATPGConfig()
    model, test_metrics = train_trba_tpg(cfg)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from __main__ import (
    final_train_records,
    final_val_records,
    final_test_records,
    TRBADataset,
    AttentionCollate,
    build_trba_records,
    trba_label_encoder,
    trba_converter,
    build_pretrained_trba,
    train_pretrained_trba,
    evaluate_attn,
)

from tpgsr.trba_tpg_config import TRBATPGConfig


class CleanTRBADataset(TRBADataset):
    """Variant of TRBADataset that reads the clean crop instead of the degraded one."""

    def __getitem__(self, idx):
        record = self.records[idx]
        # Use clean_crop_path if present (from benchmark manifest), else fall back
        img_path = record.get('clean_crop_path', record['crop_path'])
        text = record['text']

        import cv2
        crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if crop is None:
            return None

        if self.cfg.IMG_CHANNELS == 1:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop = self._resize_keep_ratio_pad(crop)

        from PIL import Image
        pil_img = Image.fromarray(crop)
        if self.pil_transform:
            pil_img = self.pil_transform(pil_img)
        img_t = self.to_tensor(pil_img)
        label = self.label_encoder.encode(text)
        return img_t, label, len(label)


def build_clean_trba_loaders(cfg):
    """Build train/val/test loaders that read CLEAN crops."""
    train_records = build_trba_records(final_train_records, cfg.BATCH_MAX_LENGTH)
    val_records = build_trba_records(final_val_records, cfg.BATCH_MAX_LENGTH)
    test_records = build_trba_records(final_test_records, cfg.BATCH_MAX_LENGTH)

    train_augment = cfg.AUGMENTATION is not None
    print(f'Train augmentation: {"ON" if train_augment else "OFF"}')

    train_ds = CleanTRBADataset(train_records, trba_label_encoder, cfg, augment=train_augment)
    val_ds = CleanTRBADataset(val_records, trba_label_encoder, cfg, augment=False)
    test_ds = CleanTRBADataset(test_records, trba_label_encoder, cfg, augment=False)

    collate = AttentionCollate(trba_label_encoder, trba_converter, cfg.BATCH_MAX_LENGTH)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate, pin_memory=True,
    )

    print(f'TRBA-TPG train batches: {len(train_loader)}')
    print(f'TRBA-TPG val batches:   {len(val_loader)}')
    print(f'TRBA-TPG test batches:  {len(test_loader)}')
    return train_loader, val_loader, test_loader


def train_trba_tpg(cfg=None):
    """Run Phase 0 for TRBA-TPG end-to-end.

    Returns (trained_model, test_metrics).
    """
    cfg = cfg or TRBATPGConfig()

    print('=' * 78)
    print('Phase 0 — Training TRBA-TPG on CLEAN crops')
    print('=' * 78)
    print(f'Epochs:       {cfg.NUM_EPOCHS}')
    print(f'Batch size:   {cfg.BATCH_SIZE}')
    print(f'Base LR:      {cfg.LR}')
    print(f'LR step:      {cfg.LR_STEP} (gamma={cfg.LR_GAMMA})')
    print(f'Weight decay: {cfg.WEIGHT_DECAY}')
    print(f'Augment:      {"ON" if cfg.AUGMENTATION is not None else "OFF"}')
    print(f'Save path:    {cfg.SAVE_PATH}')
    print(f'Device:       {cfg.DEVICE}')
    print('-' * 78)

    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)

    train_loader, val_loader, test_loader = build_clean_trba_loaders(cfg)

    model = build_pretrained_trba(trba_converter, cfg)
    train_pretrained_trba(model, train_loader, val_loader, cfg, trba_converter)

    # Reload best
    ckpt = torch.load(cfg.SAVE_PATH, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(
        f"\nLoaded best TRBA-TPG checkpoint: epoch {ckpt['epoch']}, "
        f"val_acc={ckpt['val_acc']:.1f}%, val_cer={ckpt['val_cer']:.1f}%"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=trba_converter.go_idx)
    test_metrics = evaluate_attn(model, test_loader, criterion, cfg.DEVICE, trba_converter, cfg)
    print('=' * 78)
    print(
        f"TRBA-TPG on CLEAN test -> "
        f"Word Acc: {test_metrics['word_acc']:.1f}%, "
        f"CER: {test_metrics['cer']:.1f}%, "
        f"Loss: {test_metrics['loss']:.4f}"
    )
    print('=' * 78)
    print('Phase 0 complete. This checkpoint will be the frozen TPG in Phase 1.')

    return model, test_metrics
