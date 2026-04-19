"""
Phase 4 — Final evaluation of TPGSR pipeline.

Loads the fine-tuned CRNN checkpoint (from Phase 3) and evaluates it on the
SR test cache. Compares to the baseline CRNN's heavy test accuracy.
"""

import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from __main__ import (
    cfg as baseline_cfg,
    label_encoder,
    ClovaaiCRNN,
    collate_fn,
    prepare_crnn_records,
    evaluate,
)

from tpgsr.finetune_config import FinetuneConfig
from tpgsr.sr_crnn_dataset import SRCRNNDataset


def evaluate_tpgsr_pipeline(ft_cfg=None, baseline_heavy_acc=None):
    """Load fine-tuned CRNN, evaluate on SR test cache.

    Args:
        ft_cfg: FinetuneConfig. Defaults to FinetuneConfig() — same save path
            used in Phase 3.
        baseline_heavy_acc: optional float. If provided, prints the delta.

    Returns:
        dict with test metrics.
    """
    ft_cfg = ft_cfg or FinetuneConfig()

    print('=' * 78)
    print('Phase 4 — Evaluating CRNN-TPGSR pipeline on SR test cache')
    print('=' * 78)
    print(f'Fine-tuned checkpoint: {ft_cfg.SAVE_PATH}')
    print(f'SR test cache:         {ft_cfg.SR_CACHE_ROOT}/test')
    print('-' * 78)

    # Load test records from Phase 2 cache
    with open(f'{ft_cfg.SR_CACHE_ROOT}/test/manifest.pkl', 'rb') as f:
        sr_test = pickle.load(f)
    n_sr = sum(1 for r in sr_test if r.get('sr_applied'))
    n_pass = len(sr_test) - n_sr
    print(f'SR test records: {len(sr_test)}  ({n_sr} SR / {n_pass} passthrough)')

    # Normalize + dataset
    test_norm = prepare_crnn_records(sr_test, 'SR-CRNN test')
    test_ds = SRCRNNDataset(test_norm, label_encoder, baseline_cfg, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=ft_cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=ft_cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build CRNN, load fine-tuned weights
    model = ClovaaiCRNN(
        num_classes=label_encoder.num_classes,
        input_channel=baseline_cfg.IMG_CHANNELS,
        output_channel=baseline_cfg.CNN_OUT_CHANNELS,
        hidden_size=baseline_cfg.RNN_HIDDEN_SIZE,
    ).to(ft_cfg.DEVICE)
    ckpt = torch.load(ft_cfg.SAVE_PATH, map_location=ft_cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(
        f"Loaded fine-tuned CRNN: epoch {ckpt['epoch']}, "
        f"val_acc={ckpt['val_acc']:.1f}%, val_cer={ckpt['val_cer']:.1f}%"
    )

    # Eval
    criterion = nn.CTCLoss(blank=baseline_cfg.BLANK_IDX, reduction='mean', zero_infinity=True)
    test_metrics = evaluate(model, test_loader, criterion, ft_cfg.DEVICE, label_encoder)

    print('=' * 78)
    print('FINAL CRNN-TPGSR RESULT on heavy test set')
    print('=' * 78)
    print(f"  Word accuracy: {test_metrics['word_acc']:.1f}%")
    print(f"  CER:           {test_metrics['cer']:.1f}%")
    print(f"  Loss:          {test_metrics['loss']:.4f}")

    if baseline_heavy_acc is not None:
        delta = test_metrics['word_acc'] - baseline_heavy_acc
        print('-' * 78)
        print(f'  Baseline (heavy-trained CRNN): {baseline_heavy_acc:.1f}%')
        print(f'  TPGSR (fine-tuned on SR):      {test_metrics["word_acc"]:.1f}%')
        print(f'  Delta:                         {delta:+.1f} points')
        if delta >= 5.0:
            print('  Verdict:  STRONG GAIN — proceed to TRBA-TPGSR')
        elif delta >= 2.0:
            print('  Verdict:  MODEST GAIN — within success threshold')
        elif delta >= 0.0:
            print('  Verdict:  MARGINAL — consider retraining TSRN with higher CTC')
        else:
            print('  Verdict:  NEGATIVE — pipeline needs debugging')

    print('=' * 78)
    return test_metrics
