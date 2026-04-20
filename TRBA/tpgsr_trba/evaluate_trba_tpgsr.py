"""
Phase 4 — Final evaluation of TRBA-TPGSR pipeline.

Loads fine-tuned TRBA checkpoint, evaluates on SR test cache.
"""

import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from __main__ import (
    ClovaaiTRBA,
    trba_label_encoder,
    trba_converter,
    AttentionCollate,
    build_trba_records,
    evaluate_attn,
)

from tpgsr.finetune_trba_config import FinetuneTRBAConfig
from tpgsr.sr_trba_dataset import SRTRBADataset


def evaluate_trba_tpgsr_pipeline(cfg=None, baseline_heavy_acc=None):
    cfg = cfg or FinetuneTRBAConfig()

    print('=' * 78)
    print('Phase 4 — Evaluating TRBA-TPGSR pipeline on SR test cache')
    print('=' * 78)
    print(f'Fine-tuned checkpoint: {cfg.SAVE_PATH}')
    print(f'SR test cache:         {cfg.SR_CACHE_ROOT}/test')
    print('-' * 78)

    with open(f'{cfg.SR_CACHE_ROOT}/test/manifest.pkl', 'rb') as f:
        sr_test = pickle.load(f)
    n_sr = sum(1 for r in sr_test if r.get('sr_applied'))
    print(f'SR test records: {len(sr_test)}  ({n_sr} SR / {len(sr_test) - n_sr} passthrough)')

    test_norm = build_trba_records(sr_test, cfg.BATCH_MAX_LENGTH)
    test_ds = SRTRBADataset(test_norm, trba_label_encoder, cfg, augment=False)
    collate = AttentionCollate(trba_label_encoder, trba_converter, cfg.BATCH_MAX_LENGTH)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate, pin_memory=True,
    )

    model = ClovaaiTRBA(num_class=len(trba_converter.character), cfg=cfg).to(cfg.DEVICE)
    ckpt = torch.load(cfg.SAVE_PATH, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(
        f"Loaded fine-tuned TRBA: epoch {ckpt['epoch']}, "
        f"val_acc={ckpt['val_acc']:.1f}%, val_cer={ckpt['val_cer']:.1f}%"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=trba_converter.go_idx)
    metrics = evaluate_attn(model, test_loader, criterion, cfg.DEVICE, trba_converter, cfg)

    print('=' * 78)
    print('FINAL TRBA-TPGSR RESULT on heavy test set')
    print('=' * 78)
    print(f"  Word accuracy: {metrics['word_acc']:.1f}%")
    print(f"  CER:           {metrics['cer']:.1f}%")
    print(f"  Loss:          {metrics['loss']:.4f}")

    if baseline_heavy_acc is not None:
        delta = metrics['word_acc'] - baseline_heavy_acc
        print('-' * 78)
        print(f'  Baseline (heavy-trained TRBA): {baseline_heavy_acc:.1f}%')
        print(f'  TPGSR (fine-tuned on SR):      {metrics["word_acc"]:.1f}%')
        print(f'  Delta:                         {delta:+.1f} points')
        if delta >= 5.0:
            print('  Verdict:  STRONG GAIN')
        elif delta >= 2.0:
            print('  Verdict:  MODEST GAIN')
        elif delta >= 0.0:
            print('  Verdict:  MARGINAL')
        else:
            print('  Verdict:  NEGATIVE — debug needed')

    print('=' * 78)
    return metrics
