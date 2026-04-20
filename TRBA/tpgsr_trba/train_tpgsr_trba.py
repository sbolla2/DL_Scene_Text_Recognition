"""
Phase 1 — TRBA-TPGSR training driver.

Trains TSRN with frozen TRBA-TPG. Joint loss:
    L = L1(SR, HR) + MSE(SR, HR) + CE_WEIGHT * CE(TRBA(SR, gt_text), gt_text)
applied at each refinement iteration, averaged.

Uses the same TSRN architecture as CRNN-TPGSR — architecture is recognizer-agnostic.
Only the text loss and prior injection change.
"""

import os
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from __main__ import (
    final_train_records,
    final_val_records,
    trba_label_encoder,
    trba_converter,
    build_trba_records,
)

from tpgsr.tpgsr_trba_config import TPGSRTRBAConfig
from tpgsr.tsrn import TSRN
from tpgsr.tpgsr_trba_model import TPGSRTRBA
from tpgsr.tpgsr_dataset import (
    TPGSRDataset,
    tpgsr_collate_fn,
    filter_records_by_width,
)
from tpgsr.load_trba_tpg import load_frozen_trba_tpg


def _encode_for_trba(texts, cfg, device):
    """Encode text list via trba_converter and move to device."""
    encoded, _ = trba_converter.encode(texts, batch_max_length=cfg.BATCH_MAX_LENGTH)
    return encoded.to(device)


def compute_pixel_losses(sr, hr):
    l1 = F.l1_loss(sr, hr)
    mse = F.mse_loss(sr, hr)
    return l1, mse


def compute_ce_loss(logits, targets, go_idx):
    """CrossEntropy on TRBA's (B, T, C) output against shifted targets.

    logits: (B, T, C) — teacher-forced decoder output
    targets: (B, T+1) — encoded text (we use targets[:, 1:] as the actual targets)
    """
    targets_shifted = targets[:, 1:]  # skip the leading [GO] placeholder
    # Trim to match logits length
    t = logits.size(1)
    targets_shifted = targets_shifted[:, :t]
    criterion = nn.CrossEntropyLoss(ignore_index=go_idx)
    return criterion(
        logits.contiguous().view(-1, logits.size(-1)),
        targets_shifted.contiguous().view(-1),
    )


def compute_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    if mse.item() <= 1e-12:
        return 100.0
    return 10.0 * math.log10(4.0 / mse.item())


@torch.no_grad()
def trba_word_accuracy_on_sr(logits, gt_texts):
    """Greedy-decode TRBA's inference-mode output and compare to GT texts."""
    _, preds_index = logits.max(2)
    pred_texts = trba_converter.decode(preds_index)
    correct = sum(1 for p, g in zip(pred_texts, gt_texts) if p == g)
    return correct / max(len(gt_texts), 1)


def run_epoch(model, loader, optimizer, cfg, is_train=True):
    model.train(is_train)
    # Keep TRBA frozen: eval mode for BN/Dropout, but RNN modules in train for cuDNN backward
    model.tpg_adapter.trba.eval()
    for m in model.tpg_adapter.trba.modules():
        if isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.RNN)):
            m.train()

    loss_accum = {'total': 0.0, 'l1': 0.0, 'mse': 0.0, 'ce': 0.0}
    psnr_accum = 0.0
    tpg_acc_accum = 0.0
    sample_count = 0
    last_log_time = time.time()

    device = cfg.DEVICE

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        lr, hr, _labels, _lengths, texts = batch
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        # Encode targets in TRBA's format
        trba_targets = _encode_for_trba(texts, cfg, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(lr, text_gt=trba_targets)
            sr_iters = out['sr_iterations']
            tf_iters = out['teacher_forced_iterations']

            l1_sum = mse_sum = ce_sum = 0.0
            for sr, tf_logits in zip(sr_iters, tf_iters):
                l1, mse = compute_pixel_losses(sr, hr)
                ce = compute_ce_loss(tf_logits, trba_targets, cfg.GO_IDX)
                l1_sum = l1_sum + l1
                mse_sum = mse_sum + mse
                ce_sum = ce_sum + ce
            n = len(sr_iters)
            l1_avg, mse_avg, ce_avg = l1_sum / n, mse_sum / n, ce_sum / n
            total_loss = (
                cfg.L1_WEIGHT * l1_avg
                + cfg.MSE_WEIGHT * mse_avg
                + cfg.CE_WEIGHT * ce_avg
            )

        if is_train:
            total_loss.backward()
            if cfg.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        with torch.no_grad():
            psnr = compute_psnr(out['final_sr'], hr)
            tpg_acc = trba_word_accuracy_on_sr(out['prior_iterations'][-1], texts)

        bs = lr.size(0)
        loss_accum['total'] += total_loss.item() * bs
        loss_accum['l1'] += l1_avg.item() * bs
        loss_accum['mse'] += mse_avg.item() * bs
        loss_accum['ce'] += ce_avg.item() * bs
        psnr_accum += psnr * bs
        tpg_acc_accum += tpg_acc * bs
        sample_count += bs

        if is_train and cfg.LOG_INTERVAL_BATCHES > 0 and batch_idx > 0 \
                and batch_idx % cfg.LOG_INTERVAL_BATCHES == 0:
            elapsed = time.time() - last_log_time
            rate = cfg.LOG_INTERVAL_BATCHES / max(elapsed, 1e-6)
            print(
                f'  [batch {batch_idx}/{len(loader)}] '
                f'total={total_loss.item():.4f}  '
                f'L1={l1_avg.item():.4f}  MSE={mse_avg.item():.4f}  '
                f'CE={ce_avg.item():.4f}  '
                f'PSNR={psnr:.2f}  TRBA_acc={tpg_acc*100:.1f}%  '
                f'({rate:.1f} batch/s)'
            )
            last_log_time = time.time()

    n = max(sample_count, 1)
    return {
        'total': loss_accum['total'] / n,
        'l1': loss_accum['l1'] / n,
        'mse': loss_accum['mse'] / n,
        'ce': loss_accum['ce'] / n,
        'psnr': psnr_accum / n,
        'tpg_acc': tpg_acc_accum / n,
    }


def train_tpgsr_trba(cfg=None):
    """Phase 1 entry point for TRBA variant."""
    cfg = cfg or TPGSRTRBAConfig()

    print('=' * 78)
    print('Phase 1 — Training TRBA-TPGSR (TSRN + frozen TRBA-TPG)')
    print('=' * 78)
    print(f'LR -> HR:           {cfg.LR_HEIGHT}x{cfg.LR_WIDTH} -> {cfg.HR_HEIGHT}x{cfg.HR_WIDTH}')
    print(f'Input channels:     {cfg.IN_CHANNELS} (RGB + polygon mask)')
    print(f'Iterations:         {cfg.NUM_ITERATIONS}')
    print(f'Loss weights:       L1={cfg.L1_WEIGHT} MSE={cfg.MSE_WEIGHT} CE={cfg.CE_WEIGHT}')
    print(f'Epochs / batch:     {cfg.NUM_EPOCHS} / {cfg.BATCH_SIZE}')
    print(f'LR schedule:        {cfg.LR} -> {cfg.LR_MIN} (cosine)')
    print(f'Width filter:       [{cfg.MIN_TRAIN_WIDTH}, {cfg.MAX_TRAIN_WIDTH}]')
    print(f'TRBA-TPG:           {cfg.TPG_CHECKPOINT_PATH}')
    print(f'Save path:          {cfg.SAVE_PATH}')
    print(f'Device:             {cfg.DEVICE}')
    print('-' * 78)

    # Normalize TRBA text + width filter
    print('Normalizing TRBA text + width filter...')
    train_norm = build_trba_records(final_train_records, cfg.BATCH_MAX_LENGTH)
    val_norm = build_trba_records(final_val_records, cfg.BATCH_MAX_LENGTH)

    train_filtered = filter_records_by_width(train_norm, cfg.MIN_TRAIN_WIDTH, cfg.MAX_TRAIN_WIDTH)
    val_filtered = filter_records_by_width(val_norm, cfg.MIN_TRAIN_WIDTH, cfg.MAX_TRAIN_WIDTH)
    print(f'  Train: {len(train_filtered)} / {len(train_norm)} records in width range')
    print(f'  Val:   {len(val_filtered)} / {len(val_norm)} records in width range')

    # Use the generic TPGSRDataset — it yields text that we re-encode for TRBA per-batch
    train_ds = TPGSRDataset(train_filtered, trba_label_encoder, cfg)
    val_ds = TPGSRDataset(val_filtered, trba_label_encoder, cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=tpgsr_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=tpgsr_collate_fn,
        pin_memory=True,
    )
    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    # Load frozen TRBA-TPG
    print('Loading frozen TRBA-TPG...')
    trba = load_frozen_trba_tpg(cfg.TPG_CHECKPOINT_PATH, cfg)

    prior_feat_channels = 32

    tsrn = TSRN(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        hidden_channels=cfg.TSRN_HIDDEN_CHANNELS,
        num_blocks=cfg.TSRN_NUM_BLOCKS,
        scale=cfg.SR_SCALE,
        use_stn=cfg.TSRN_STN,
        prior_channels=prior_feat_channels,
    ).to(cfg.DEVICE)

    model = TPGSRTRBA(
        tsrn=tsrn,
        trba=trba,
        num_classes=cfg.TPG_NUM_CLASSES,
        num_iterations=cfg.NUM_ITERATIONS,
        prior_feat_channels=prior_feat_channels,
        lr_spatial_size=(cfg.LR_HEIGHT, cfg.LR_WIDTH),
        trba_image_size=(cfg.TRBA_IMG_HEIGHT, cfg.TRBA_IMG_WIDTH),
        batch_max_length=cfg.BATCH_MAX_LENGTH,
    ).to(cfg.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model total params:     {total:,}')
    print(f'Model trainable params: {trainable:,}  (TRBA is frozen)')

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR, betas=cfg.BETAS, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR_MIN,
    )

    history = {
        'train_total': [], 'train_l1': [], 'train_mse': [], 'train_ce': [],
        'train_psnr': [], 'train_tpg_acc': [],
        'val_total': [], 'val_l1': [], 'val_mse': [], 'val_ce': [],
        'val_psnr': [], 'val_tpg_acc': [], 'lr': [],
    }
    best_val_tpg_acc = -1.0

    print('=' * 78)
    header = (
        f"{'Ep':>3} | {'TrTot':>6} {'TrL1':>6} {'TrMSE':>6} {'TrCE':>6} "
        f"{'TrPSNR':>6} {'TrTRBA':>7} | "
        f"{'VaTot':>6} {'VaL1':>6} {'VaMSE':>6} {'VaCE':>6} "
        f"{'VaPSNR':>6} {'VaTRBA':>7} | {'LR':>9} | {'Time':>5}"
    )
    print(header)
    print('-' * len(header))

    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, cfg, is_train=True)
        val_m = run_epoch(model, val_loader, optimizer, cfg, is_train=False)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        for split, m in [('train', train_m), ('val', val_m)]:
            history[f'{split}_total'].append(m['total'])
            history[f'{split}_l1'].append(m['l1'])
            history[f'{split}_mse'].append(m['mse'])
            history[f'{split}_ce'].append(m['ce'])
            history[f'{split}_psnr'].append(m['psnr'])
            history[f'{split}_tpg_acc'].append(m['tpg_acc'])
        history['lr'].append(current_lr)

        print(
            f'{epoch:3d} | '
            f'{train_m["total"]:6.3f} {train_m["l1"]:6.3f} '
            f'{train_m["mse"]:6.3f} {train_m["ce"]:6.3f} '
            f'{train_m["psnr"]:6.2f} {train_m["tpg_acc"]*100:6.1f} | '
            f'{val_m["total"]:6.3f} {val_m["l1"]:6.3f} '
            f'{val_m["mse"]:6.3f} {val_m["ce"]:6.3f} '
            f'{val_m["psnr"]:6.2f} {val_m["tpg_acc"]*100:6.1f} | '
            f'{current_lr:.2e} | {elapsed:4.0f}s'
        )

        if val_m['tpg_acc'] > best_val_tpg_acc:
            best_val_tpg_acc = val_m['tpg_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_tpg_acc': best_val_tpg_acc,
                'val_psnr': val_m['psnr'],
            }, cfg.SAVE_PATH)
            print(f'       ** Saved best model (val_tpg_acc={best_val_tpg_acc*100:.1f}%)')

    print('=' * 78)
    print(f'Training complete. Best val TRBA accuracy on SR: {best_val_tpg_acc*100:.1f}%')
    print(f'Checkpoint: {cfg.SAVE_PATH}')
    return model, history
