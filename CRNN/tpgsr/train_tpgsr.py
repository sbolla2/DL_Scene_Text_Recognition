"""
Phase 1 — TPGSR training driver.

Trains TSRN with frozen CRNN-TPG under the joint loss:
    L = L1(SR, HR) + MSE(SR, HR) + 0.1 * CTC(TPG(SR), GT_text)
applied at each refinement iteration and averaged.

Usage (from a Colab cell after the baseline notebook has run):
    import sys
    if '/content' not in sys.path:
        sys.path.insert(0, '/content')
    from tpgsr.train_tpgsr import train_tpgsr
    history = train_tpgsr()
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
    label_encoder,
    prepare_crnn_records,
)

from tpgsr.tpgsr_config import TPGSRConfig
from tpgsr.tsrn import TSRN
from tpgsr.tpgsr_model import TPGSR
from tpgsr.tpgsr_dataset import (
    TPGSRDataset,
    tpgsr_collate_fn,
    filter_records_by_width,
)
from tpgsr.load_tpg import load_frozen_tpg


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def compute_pixel_losses(sr, hr):
    """Return (L1, MSE) losses between SR and HR, both in [-1, 1] tanh range."""
    l1 = F.l1_loss(sr, hr)
    mse = F.mse_loss(sr, hr)
    return l1, mse


def compute_text_loss(log_probs, targets, target_lengths, blank_idx):
    """CTC loss between TPG's log-probs on SR output and GT text.

    Args:
        log_probs: (T, B, num_classes) from CRNN.
        targets: (sum of lengths,) concatenated 1-D LongTensor.
        target_lengths: (B,) LongTensor.
        blank_idx: CTC blank class (0 in our charset).
    """
    t_steps, b, _ = log_probs.shape
    input_lengths = torch.full((b,), t_steps, dtype=torch.long, device=log_probs.device)
    ctc = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    return ctc(log_probs, targets, input_lengths, target_lengths)


def compute_psnr(sr, hr):
    """PSNR in dB between (B, C, H, W) tensors in [-1, 1] range."""
    mse = F.mse_loss(sr, hr)
    # Signal range is 2.0 (from -1 to 1), so max^2 = 4
    if mse.item() <= 1e-12:
        return 100.0
    return 10.0 * math.log10(4.0 / mse.item())


# ---------------------------------------------------------------------------
# TPG monitoring metric
# ---------------------------------------------------------------------------
@torch.no_grad()
def tpg_word_accuracy_on_sr(log_probs, gt_texts, label_encoder):
    """Greedy-decode the CRNN TPG's prediction on SR output and compare to GT."""
    _, preds = log_probs.max(2)         # (T, B)
    preds = preds.permute(1, 0)         # (B, T)

    correct = 0
    for i in range(preds.size(0)):
        raw = preds[i].cpu().tolist()
        text = label_encoder.decode(raw, remove_duplicates=True, remove_blank=True)
        if text == gt_texts[i]:
            correct += 1
    return correct / max(len(gt_texts), 1)


# ---------------------------------------------------------------------------
# Training / evaluation passes
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, cfg, is_train=True):
    """One pass over loader. Returns dict of average metrics."""
    model.train(is_train)
    # TPG parameter handling:
    # - Params are already frozen via requires_grad=False (set in load_frozen_tpg).
    # - BatchNorm/Dropout layers should stay in eval mode so TPG's running stats
    #   don't drift from what it was trained on.
    # - RNN (LSTM) layers MUST be in train mode, because cuDNN's RNN backward
    #   refuses to run through eval-mode RNNs — even when we only want gradients
    #   to flow THROUGH (not into) the RNN's weights.
    model.tpg_adapter.crnn.eval()
    for m in model.tpg_adapter.crnn.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            m.train()

    loss_accum = {'total': 0.0, 'l1': 0.0, 'mse': 0.0, 'ctc': 0.0}
    psnr_accum = 0.0
    tpg_acc_accum = 0.0
    batch_count = 0
    sample_count = 0

    device = cfg.DEVICE

    last_log_time = time.time()

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        lr, hr, labels, lengths, texts = batch
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(lr)
            sr_iterations = out['sr_iterations']
            prior_iterations = out['prior_iterations']

            # Accumulate losses across iterations.
            l1_sum, mse_sum, ctc_sum = 0.0, 0.0, 0.0
            for sr, log_probs in zip(sr_iterations, prior_iterations):
                l1, mse = compute_pixel_losses(sr, hr)
                ctc = compute_text_loss(log_probs, labels, lengths, cfg.BLANK_IDX)
                l1_sum = l1_sum + l1
                mse_sum = mse_sum + mse
                ctc_sum = ctc_sum + ctc
            n_iter = len(sr_iterations)
            l1_avg = l1_sum / n_iter
            mse_avg = mse_sum / n_iter
            ctc_avg = ctc_sum / n_iter

            total_loss = (
                cfg.L1_WEIGHT * l1_avg
                + cfg.MSE_WEIGHT * mse_avg
                + cfg.CTC_WEIGHT * ctc_avg
            )

        if is_train:
            total_loss.backward()
            if cfg.GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        # Metrics from the final iteration (most representative)
        with torch.no_grad():
            psnr = compute_psnr(out['final_sr'], hr)
            tpg_acc = tpg_word_accuracy_on_sr(
                out['prior_iterations'][-1], texts, label_encoder
            )

        bs = lr.size(0)
        loss_accum['total'] += total_loss.item() * bs
        loss_accum['l1'] += l1_avg.item() * bs
        loss_accum['mse'] += mse_avg.item() * bs
        loss_accum['ctc'] += ctc_avg.item() * bs
        psnr_accum += psnr * bs
        tpg_acc_accum += tpg_acc * bs
        sample_count += bs
        batch_count += 1

        # Periodic within-epoch log
        if is_train and cfg.LOG_INTERVAL_BATCHES > 0 and batch_idx > 0 \
                and batch_idx % cfg.LOG_INTERVAL_BATCHES == 0:
            elapsed = time.time() - last_log_time
            batches_per_sec = cfg.LOG_INTERVAL_BATCHES / max(elapsed, 1e-6)
            print(
                f'  [batch {batch_idx}/{len(loader)}] '
                f'total={total_loss.item():.4f}  '
                f'L1={l1_avg.item():.4f}  MSE={mse_avg.item():.4f}  '
                f'CTC={ctc_avg.item():.4f}  '
                f'PSNR={psnr:.2f}  TPG_acc={tpg_acc*100:.1f}%  '
                f'({batches_per_sec:.1f} batch/s)'
            )
            last_log_time = time.time()

    n = max(sample_count, 1)
    return {
        'total': loss_accum['total'] / n,
        'l1': loss_accum['l1'] / n,
        'mse': loss_accum['mse'] / n,
        'ctc': loss_accum['ctc'] / n,
        'psnr': psnr_accum / n,
        'tpg_acc': tpg_acc_accum / n,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def train_tpgsr(cfg=None):
    """Run Phase 1 end to end.

    Returns the trained model and a history dict.
    """
    cfg = cfg or TPGSRConfig()

    print('=' * 78)
    print('Phase 1 — Training TPGSR (TSRN + frozen CRNN-TPG)')
    print('=' * 78)
    print(f'LR -> HR:           {cfg.LR_HEIGHT}x{cfg.LR_WIDTH} -> {cfg.HR_HEIGHT}x{cfg.HR_WIDTH}')
    print(f'Input channels:     {cfg.IN_CHANNELS} (RGB + polygon mask)')
    print(f'Iterations:         {cfg.NUM_ITERATIONS}')
    print(f'Loss weights:       L1={cfg.L1_WEIGHT} MSE={cfg.MSE_WEIGHT} CTC={cfg.CTC_WEIGHT}')
    print(f'Epochs / batch:     {cfg.NUM_EPOCHS} / {cfg.BATCH_SIZE}')
    print(f'LR schedule:        {cfg.LR} -> {cfg.LR_MIN} (cosine)')
    print(f'Width filter:       [{cfg.MIN_TRAIN_WIDTH}, {cfg.MAX_TRAIN_WIDTH}]')
    print(f'TPG checkpoint:     {cfg.TPG_CHECKPOINT_PATH}')
    print(f'Save path:          {cfg.SAVE_PATH}')
    print(f'Device:             {cfg.DEVICE}')
    print('-' * 78)

    # --- Normalize text (charset filter + length filter) FIRST ---
    # Same normalization Phase 0 uses so label_encoder.encode() never sees
    # out-of-charset characters.
    print('Normalizing text (charset filter + length filter)...')
    train_normalized = prepare_crnn_records(final_train_records, 'TPGSR train')
    val_normalized = prepare_crnn_records(final_val_records, 'TPGSR val')

    # --- Width-filter records ---
    print('Filtering records by width...')
    train_filtered = filter_records_by_width(
        train_normalized, cfg.MIN_TRAIN_WIDTH, cfg.MAX_TRAIN_WIDTH
    )
    val_filtered = filter_records_by_width(
        val_normalized, cfg.MIN_TRAIN_WIDTH, cfg.MAX_TRAIN_WIDTH
    )
    print(f'  Train: {len(train_filtered)} / {len(train_normalized)} records in width range')
    print(f'  Val:   {len(val_filtered)} / {len(val_normalized)} records in width range')

    if len(train_filtered) == 0 or len(val_filtered) == 0:
        raise RuntimeError(
            f'No records fell in the [{cfg.MIN_TRAIN_WIDTH}, {cfg.MAX_TRAIN_WIDTH}] '
            'width range. Check your crop_bbox metadata or loosen the filter.'
        )

    # --- Datasets / loaders ---
    train_ds = TPGSRDataset(train_filtered, label_encoder, cfg)
    val_ds = TPGSRDataset(val_filtered, label_encoder, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=tpgsr_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=tpgsr_collate_fn,
        pin_memory=True,
    )
    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    # --- Build model ---
    print('Loading frozen TPG...')
    tpg = load_frozen_tpg(cfg.TPG_CHECKPOINT_PATH, cfg)

    # TSRN needs prior_channels == prior_feat_channels (wrapper asserts this)
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

    model = TPGSR(
        tsrn=tsrn,
        tpg=tpg,
        num_classes=cfg.TPG_NUM_CLASSES,
        num_iterations=cfg.NUM_ITERATIONS,
        prior_feat_channels=prior_feat_channels,
        lr_spatial_size=(cfg.LR_HEIGHT, cfg.LR_WIDTH),
    ).to(cfg.DEVICE)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'TPGSR total params:     {total:,}')
    print(f'TPGSR trainable params: {trainable:,}  (TPG is frozen)')

    # --- Optimizer / scheduler ---
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.LR,
        betas=cfg.BETAS,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR_MIN
    )

    # --- Training loop ---
    history = {
        'train_total': [], 'train_l1': [], 'train_mse': [], 'train_ctc': [],
        'train_psnr': [], 'train_tpg_acc': [],
        'val_total': [], 'val_l1': [], 'val_mse': [], 'val_ctc': [],
        'val_psnr': [], 'val_tpg_acc': [], 'lr': [],
    }
    best_val_tpg_acc = -1.0

    print('=' * 78)
    header = (
        f"{'Ep':>3} | {'TrTot':>6} {'TrL1':>6} {'TrMSE':>6} {'TrCTC':>6} "
        f"{'TrPSNR':>6} {'TrTPG':>6} | "
        f"{'VaTot':>6} {'VaL1':>6} {'VaMSE':>6} {'VaCTC':>6} "
        f"{'VaPSNR':>6} {'VaTPG':>6} | {'LR':>9} | {'Time':>5}"
    )
    print(header)
    print('-' * len(header))

    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, optimizer, cfg, is_train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, cfg, is_train=False)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        elapsed = time.time() - t0

        for split, m in [('train', train_metrics), ('val', val_metrics)]:
            history[f'{split}_total'].append(m['total'])
            history[f'{split}_l1'].append(m['l1'])
            history[f'{split}_mse'].append(m['mse'])
            history[f'{split}_ctc'].append(m['ctc'])
            history[f'{split}_psnr'].append(m['psnr'])
            history[f'{split}_tpg_acc'].append(m['tpg_acc'])
        history['lr'].append(current_lr)

        print(
            f'{epoch:3d} | '
            f'{train_metrics["total"]:6.3f} {train_metrics["l1"]:6.3f} '
            f'{train_metrics["mse"]:6.3f} {train_metrics["ctc"]:6.3f} '
            f'{train_metrics["psnr"]:6.2f} {train_metrics["tpg_acc"]*100:6.1f} | '
            f'{val_metrics["total"]:6.3f} {val_metrics["l1"]:6.3f} '
            f'{val_metrics["mse"]:6.3f} {val_metrics["ctc"]:6.3f} '
            f'{val_metrics["psnr"]:6.2f} {val_metrics["tpg_acc"]*100:6.1f} | '
            f'{current_lr:.2e} | {elapsed:4.0f}s'
        )

        # Save best by val TPG accuracy (text-readability is the downstream metric)
        if val_metrics['tpg_acc'] > best_val_tpg_acc:
            best_val_tpg_acc = val_metrics['tpg_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_tpg_acc': best_val_tpg_acc,
                'val_psnr': val_metrics['psnr'],
                'cfg': {k: v for k, v in vars(cfg).items() if not k.startswith('_')},
            }, cfg.SAVE_PATH)
            print(f'       ** Saved best model (val_tpg_acc={best_val_tpg_acc*100:.1f}%)')

    print('=' * 78)
    print(f'Training complete. Best val TPG accuracy: {best_val_tpg_acc*100:.1f}%')
    print(f'Checkpoint: {cfg.SAVE_PATH}')

    return model, history