"""
Phase 3 — Fine-tune heavy-trained TRBA on SR cache.

Continues training from trba_totaltext_heavy_best.pth.
Goal: adapt TRBA from raw-degraded to SR-output distribution.
"""

import os
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from __main__ import (
    ClovaaiTRBA,
    trba_label_encoder,
    trba_converter,
    AttentionCollate,
    build_trba_records,
    train_one_epoch_attn,
    evaluate_attn,
)

from tpgsr.finetune_trba_config import FinetuneTRBAConfig
from tpgsr.sr_trba_dataset import SRTRBADataset


def _load_sr_records(sr_root):
    with open(f'{sr_root}/train/manifest.pkl', 'rb') as f:
        train_records = pickle.load(f)
    with open(f'{sr_root}/val/manifest.pkl', 'rb') as f:
        val_records = pickle.load(f)
    with open(f'{sr_root}/test/manifest.pkl', 'rb') as f:
        test_records = pickle.load(f)
    return train_records, val_records, test_records


def _build_trba_from_checkpoint(checkpoint_path, cfg):
    model = ClovaaiTRBA(num_class=len(trba_converter.character), cfg=cfg).to(cfg.DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(
        f"Loaded baseline TRBA: epoch {ckpt['epoch']}, "
        f"val_acc={ckpt['val_acc']:.1f}%, val_cer={ckpt['val_cer']:.1f}%"
    )
    return model


def finetune_trba_on_sr(cfg=None):
    cfg = cfg or FinetuneTRBAConfig()

    print('=' * 78)
    print('Phase 3 — Fine-tuning TRBA on SR cache')
    print('=' * 78)
    print(f'Baseline:   {cfg.BASELINE_CHECKPOINT_PATH}')
    print(f'SR cache:   {cfg.SR_CACHE_ROOT}')
    print(f'Save path:  {cfg.SAVE_PATH}')
    print(f'Epochs:     {cfg.NUM_EPOCHS}')
    print(f'LR:         {cfg.LR}')
    print(f'Device:     {cfg.DEVICE}')
    print('-' * 78)

    sr_train, sr_val, sr_test = _load_sr_records(cfg.SR_CACHE_ROOT)
    n_sr_train = sum(1 for r in sr_train if r.get('sr_applied'))
    print(
        f'SR cache loaded: train={len(sr_train)} '
        f'({n_sr_train} SR / {len(sr_train) - n_sr_train} passthrough), '
        f'val={len(sr_val)}, test={len(sr_test)}'
    )

    train_norm = build_trba_records(sr_train, cfg.BATCH_MAX_LENGTH)
    val_norm = build_trba_records(sr_val, cfg.BATCH_MAX_LENGTH)

    train_ds = SRTRBADataset(train_norm, trba_label_encoder, cfg, augment=True)
    val_ds = SRTRBADataset(val_norm, trba_label_encoder, cfg, augment=False)

    collate = AttentionCollate(trba_label_encoder, trba_converter, cfg.BATCH_MAX_LENGTH)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate, pin_memory=True,
    )

    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    model = _build_trba_from_checkpoint(cfg.BASELINE_CHECKPOINT_PATH, cfg)

    # Differential LR groups — same pattern as baseline TRBA training
    tps_params = [p for n, p in model.named_parameters() if n.startswith('Transformation')]
    feat_params = [p for n, p in model.named_parameters() if n.startswith('FeatureExtraction')]
    seq_params = [p for n, p in model.named_parameters() if n.startswith('SequenceModeling')]
    pred_params = [p for n, p in model.named_parameters() if n.startswith('Prediction')]

    optimizer = optim.Adam([
        {'params': tps_params, 'lr': cfg.LR * 0.05},
        {'params': feat_params, 'lr': cfg.LR * 0.10},
        {'params': seq_params, 'lr': cfg.LR * 0.25},
        {'params': pred_params, 'lr': cfg.LR * 0.50},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR_MIN,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=trba_converter.go_idx)

    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_cer': [], 'lr': []}
    best_val_acc = -1.0

    print('=' * 78)
    header = f"{'Ep':>3} | {'TrLoss':>7} | {'VaLoss':>7} {'VaAcc':>6} {'VaCER':>6} | {'LR':>9} | {'Time':>5}"
    print(header)
    print('-' * len(header))

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch_attn(
            model, train_loader, criterion, optimizer, cfg.DEVICE,
            trba_converter, cfg, grad_clip=cfg.GRAD_CLIP,
        )
        val_metrics = evaluate_attn(
            model, val_loader, criterion, cfg.DEVICE, trba_converter, cfg,
        )

        current_lr = optimizer.param_groups[-1]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['word_acc'])
        history['val_cer'].append(val_metrics['cer'])
        history['lr'].append(current_lr)

        print(
            f'{epoch:3d} | {train_metrics["loss"]:7.4f} | '
            f'{val_metrics["loss"]:7.4f} '
            f'{val_metrics["word_acc"]:5.1f}% '
            f'{val_metrics["cer"]:5.1f}% | '
            f'{current_lr:.2e} | {elapsed:4.0f}s'
        )

        if val_metrics['word_acc'] > best_val_acc:
            best_val_acc = val_metrics['word_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'val_cer': val_metrics['cer'],
                'val_loss': val_metrics['loss'],
            }, cfg.SAVE_PATH)
            print(f'       ** Saved best model (val_acc={best_val_acc:.1f}%)')

    print('=' * 78)
    print(f'Fine-tuning complete. Best val acc: {best_val_acc:.1f}%')
    return model, history
