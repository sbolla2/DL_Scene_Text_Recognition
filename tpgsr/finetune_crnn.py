"""
Phase 3 — Fine-tune heavy-trained CRNN on SR cache.

Continues training from crnn_totaltext_heavy_best.pth (34.3% baseline).
Goal: adapt the CRNN from raw-degraded distribution to SR-output distribution.

Uses train+val SR caches from Phase 2. Tests later in Phase 4.
"""

import os
import pickle
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from __main__ import (
    Config,
    cfg as baseline_cfg,
    label_encoder,
    ClovaaiCRNN,
    collate_fn,
    prepare_crnn_records,
    evaluate,
)

from tpgsr.finetune_config import FinetuneConfig
from tpgsr.sr_crnn_dataset import SRCRNNDataset


def _load_sr_records(sr_root):
    """Load SR cache manifests for all three splits."""
    with open(f'{sr_root}/train/manifest.pkl', 'rb') as f:
        train_records = pickle.load(f)
    with open(f'{sr_root}/val/manifest.pkl', 'rb') as f:
        val_records = pickle.load(f)
    with open(f'{sr_root}/test/manifest.pkl', 'rb') as f:
        test_records = pickle.load(f)
    return train_records, val_records, test_records


def _build_crnn_from_checkpoint(checkpoint_path, ft_cfg, base_cfg):
    """Instantiate ClovaaiCRNN matching baseline architecture, load checkpoint."""
    model = ClovaaiCRNN(
        num_classes=label_encoder.num_classes,
        input_channel=base_cfg.IMG_CHANNELS,
        output_channel=base_cfg.CNN_OUT_CHANNELS,
        hidden_size=base_cfg.RNN_HIDDEN_SIZE,
    ).to(ft_cfg.DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=ft_cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(
        f"Loaded baseline CRNN: epoch {ckpt['epoch']}, "
        f"val_acc={ckpt['val_acc']:.1f}%, val_cer={ckpt['val_cer']:.1f}%"
    )
    return model


def _set_feature_extractor_grad(model, requires_grad):
    """Toggle requires_grad on the VGG CNN backbone. Matches baseline helper."""
    for name, param in model.named_parameters():
        if name.startswith('FeatureExtraction'):
            param.requires_grad = requires_grad


def _make_param_groups(model, base_lr):
    """Differential LR like the baseline's train_with_pretrained."""
    groups = [
        {'params': [], 'lr': base_lr * 0.1, 'name': 'feature_extractor'},
        {'params': [], 'lr': base_lr * 0.5, 'name': 'sequence_model'},
        {'params': [], 'lr': base_lr * 1.0, 'name': 'prediction'},
    ]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('FeatureExtraction'):
            groups[0]['params'].append(p)
        elif name.startswith('SequenceModeling'):
            groups[1]['params'].append(p)
        else:
            groups[2]['params'].append(p)
    return groups


def finetune_crnn_on_sr(ft_cfg=None):
    """Phase 3 entry point.

    Returns (finetuned_model, history).
    """
    ft_cfg = ft_cfg or FinetuneConfig()

    print('=' * 78)
    print('Phase 3 — Fine-tuning CRNN on SR cache')
    print('=' * 78)
    print(f'Baseline:   {ft_cfg.BASELINE_CHECKPOINT_PATH}')
    print(f'SR cache:   {ft_cfg.SR_CACHE_ROOT}')
    print(f'Save path:  {ft_cfg.SAVE_PATH}')
    print(f'Epochs:     {ft_cfg.NUM_EPOCHS}')
    print(f'LR:         {ft_cfg.LR}')
    print(f'Device:     {ft_cfg.DEVICE}')
    print('-' * 78)

    # Load SR records
    sr_train, sr_val, sr_test = _load_sr_records(ft_cfg.SR_CACHE_ROOT)
    n_sr_train = sum(1 for r in sr_train if r.get('sr_applied'))
    n_pass_train = len(sr_train) - n_sr_train
    print(
        f'SR cache loaded: train={len(sr_train)} '
        f'({n_sr_train} SR / {n_pass_train} passthrough), '
        f'val={len(sr_val)}, test={len(sr_test)}'
    )

    # Normalize text (charset + length filter) — same as baseline
    train_norm = prepare_crnn_records(sr_train, 'SR-CRNN train')
    val_norm = prepare_crnn_records(sr_val, 'SR-CRNN val')

    # Datasets / loaders — train with augmentation ON to match baseline config
    train_ds = SRCRNNDataset(train_norm, label_encoder, baseline_cfg, augment=True)
    val_ds = SRCRNNDataset(val_norm, label_encoder, baseline_cfg, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=ft_cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=ft_cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ft_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=ft_cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    # Build model from baseline checkpoint
    model = _build_crnn_from_checkpoint(
        ft_cfg.BASELINE_CHECKPOINT_PATH, ft_cfg, baseline_cfg
    )

    # Start with VGG frozen for epoch 1
    if ft_cfg.FREEZE_CNN_EPOCHS > 0:
        _set_feature_extractor_grad(model, False)
        print(f'Feature extractor FROZEN for first {ft_cfg.FREEZE_CNN_EPOCHS} epoch(s)')

    # Optimizer / scheduler
    param_groups = _make_param_groups(model, ft_cfg.LR)
    optimizer = torch.optim.Adam(param_groups, weight_decay=ft_cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ft_cfg.NUM_EPOCHS, eta_min=ft_cfg.LR_MIN
    )

    criterion = nn.CTCLoss(blank=baseline_cfg.BLANK_IDX, reduction='mean', zero_infinity=True)

    # Training loop
    os.makedirs(os.path.dirname(ft_cfg.SAVE_PATH), exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_cer': [], 'lr': []}
    best_val_acc = -1.0

    print('=' * 78)
    header = f"{'Ep':>3} | {'TrLoss':>7} | {'VaLoss':>7} {'VaAcc':>6} {'VaCER':>6} | {'LR':>9} | {'Time':>5}"
    print(header)
    print('-' * len(header))

    for epoch in range(1, ft_cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        # Unfreeze feature extractor after the freeze epochs
        if epoch == ft_cfg.FREEZE_CNN_EPOCHS + 1:
            _set_feature_extractor_grad(model, True)
            # Rebuild param groups so optimizer knows about the now-trainable params
            param_groups = _make_param_groups(model, ft_cfg.LR)
            optimizer = torch.optim.Adam(param_groups, weight_decay=ft_cfg.WEIGHT_DECAY)
            # Reset scheduler to continue from current progress
            remaining_epochs = ft_cfg.NUM_EPOCHS - ft_cfg.FREEZE_CNN_EPOCHS
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining_epochs, eta_min=ft_cfg.LR_MIN
            )
            print(f'   ** Unfreezing feature extractor at epoch {epoch}')

        # Train one epoch
        model.train()
        train_loss_sum, train_samples = 0.0, 0
        for batch in train_loader:
            if batch is None:
                continue
            images, targets, lengths = batch
            images = images.to(ft_cfg.DEVICE, non_blocking=True)
            targets = targets.to(ft_cfg.DEVICE, non_blocking=True)
            lengths = lengths.to(ft_cfg.DEVICE, non_blocking=True)

            optimizer.zero_grad()
            log_probs = model(images)
            t_steps = log_probs.size(0)
            input_lengths = torch.full(
                (images.size(0),), t_steps, dtype=torch.long, device=ft_cfg.DEVICE
            )
            loss = criterion(log_probs, targets, input_lengths, lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = images.size(0)
            train_loss_sum += loss.item() * bs
            train_samples += bs

        # Val
        val_metrics = evaluate(model, val_loader, criterion, ft_cfg.DEVICE, label_encoder)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        train_loss = train_loss_sum / max(train_samples, 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['word_acc'])
        history['val_cer'].append(val_metrics['cer'])
        history['lr'].append(current_lr)

        print(
            f'{epoch:3d} | '
            f'{train_loss:7.4f} | '
            f'{val_metrics["loss"]:7.4f} '
            f'{val_metrics["word_acc"]:5.1f}% '
            f'{val_metrics["cer"]:5.1f}% | '
            f'{current_lr:.2e} | {elapsed:4.0f}s'
        )

        # Save best
        if val_metrics['word_acc'] > best_val_acc:
            best_val_acc = val_metrics['word_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc,
                'val_cer': val_metrics['cer'],
                'val_loss': val_metrics['loss'],
                'train_loss': train_loss,
            }, ft_cfg.SAVE_PATH)
            print(f'       ** Saved best model (val_acc={best_val_acc:.1f}%)')

    print('=' * 78)
    print(f'Fine-tuning complete. Best val accuracy: {best_val_acc:.1f}%')
    print(f'Checkpoint: {ft_cfg.SAVE_PATH}')

    return model, history
