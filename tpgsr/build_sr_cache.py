"""
Phase 2 — Run trained TSRN on all benchmark crops, cache SR outputs to disk.

For each split (train/val/test):
  1. Load each record's degraded crop + mask
  2. If original clean-crop width is in [MIN_CROP_WIDTH, MAX_CROP_WIDTH]:
       - Build 4-ch LR tensor
       - Run TSRN (with TPGSR wrapper for full 2-iter refinement)
       - Save SR output as PNG to SR_CACHE_ROOT/{split}/{idx}.png
       - Record['sr_path'] = output path
     Else:
       - Skip TSRN, flag record as passthrough
       - Record['sr_path'] = None
  3. Save updated records to manifest.pkl in SR_CACHE_ROOT/{split}/

Usage:
    from tpgsr.build_sr_cache import build_sr_cache
    sr_train, sr_val, sr_test = build_sr_cache()
"""

import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from __main__ import (
    final_train_records,
    final_val_records,
    final_test_records,
)

from tpgsr.sr_cache_config import SRCacheConfig
from tpgsr.tsrn import TSRN
from tpgsr.tpgsr_model import TPGSR
from tpgsr.load_tpg import load_frozen_tpg
from tpgsr.tpgsr_dataset import (
    _to_tensor_tanh,
    _to_mask_tensor,
)


def _load_tpgsr_model(cfg):
    """Instantiate TSRN + TPGSR wrapper, load checkpoint weights."""
    print(f'Loading TSRN checkpoint: {cfg.TSRN_CHECKPOINT_PATH}')

    tpg = load_frozen_tpg(cfg.TPG_CHECKPOINT_PATH, cfg)

    tsrn = TSRN(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        hidden_channels=cfg.TSRN_HIDDEN_CHANNELS,
        num_blocks=cfg.TSRN_NUM_BLOCKS,
        scale=2,
        use_stn=cfg.TSRN_STN,
        prior_channels=cfg.PRIOR_FEAT_CHANNELS,
    ).to(cfg.DEVICE)

    model = TPGSR(
        tsrn=tsrn,
        tpg=tpg,
        num_classes=cfg.TPG_NUM_CLASSES,
        num_iterations=cfg.NUM_ITERATIONS,
        prior_feat_channels=cfg.PRIOR_FEAT_CHANNELS,
        lr_spatial_size=(cfg.LR_HEIGHT, cfg.LR_WIDTH),
    ).to(cfg.DEVICE)

    # The STN has a lazily-built FC layer whose dimensions depend on the
    # first forward-pass input. Run a dummy forward so those FC parameters
    # are materialized before we load the checkpoint; otherwise the
    # checkpoint's FC keys are flagged as "unexpected".
    with torch.no_grad():
        dummy_lr = torch.zeros(
            1, cfg.IN_CHANNELS, cfg.LR_HEIGHT, cfg.LR_WIDTH,
            device=cfg.DEVICE,
        )
        _ = model(dummy_lr)

    ckpt = torch.load(cfg.TSRN_CHECKPOINT_PATH, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(
        f"  TSRN checkpoint epoch {ckpt['epoch']}, "
        f"val_tpg_acc={ckpt['val_tpg_acc']*100:.1f}%, "
        f"val_psnr={ckpt['val_psnr']:.2f} dB"
    )
    return model


def _record_width(record):
    """Width of the original clean crop (used for the hybrid passthrough decision)."""
    bbox = record.get('crop_bbox')
    if bbox:
        x1, y1, x2, y2 = bbox
        return x2 - x1
    # Fallback: read clean crop from disk
    clean_path = record.get('clean_crop_path', record['crop_path'])
    img = cv2.imread(clean_path, cv2.IMREAD_COLOR)
    if img is None:
        return -1
    return img.shape[1]


def _build_lr_batch(records, cfg):
    """Turn a list of records into a batched (B, 4, LR_H, LR_W) tensor."""
    lrs = []
    for record in records:
        lr_bgr = cv2.imread(record['crop_path'], cv2.IMREAD_COLOR)
        if lr_bgr is None:
            lrs.append(None)
            continue
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        lr_resized = cv2.resize(
            lr_rgb, (cfg.LR_WIDTH, cfg.LR_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )
        lr_tensor_rgb = _to_tensor_tanh(lr_resized)

        if record.get('mask_path') and os.path.exists(record['mask_path']):
            mask_uint8 = cv2.imread(record['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask_uint8 is None:
                mask_uint8 = np.full(
                    (cfg.LR_HEIGHT, cfg.LR_WIDTH), 255, dtype=np.uint8
                )
        else:
            mask_uint8 = np.full(
                (cfg.LR_HEIGHT, cfg.LR_WIDTH), 255, dtype=np.uint8
            )
        mask_tensor = _to_mask_tensor(
            mask_uint8, (cfg.LR_HEIGHT, cfg.LR_WIDTH)
        )

        lr_tensor = torch.cat([lr_tensor_rgb, mask_tensor], dim=0)
        lrs.append(lr_tensor)

    # Filter Nones, track which indices were valid
    valid_indices = [i for i, t in enumerate(lrs) if t is not None]
    if not valid_indices:
        return None, []
    valid_tensors = [lrs[i] for i in valid_indices]
    batch = torch.stack(valid_tensors, 0)
    return batch, valid_indices


def _save_sr_png(sr_tensor, path):
    """sr_tensor: (3, H, W) in [-1, 1]. Saves as BGR PNG."""
    sr_np = ((sr_tensor + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
    sr_np = np.transpose(sr_np, (1, 2, 0))      # (H, W, 3) RGB
    sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, sr_bgr)


def _save_sample_grid(records_with_sr, cfg, split_name, limit):
    """Save a visual grid of (degraded | SR | clean) triplets for eyeballing."""
    sample_path = os.path.join(cfg.SR_CACHE_ROOT, f'samples_{split_name}.png')
    triplets = []
    selected = 0
    for record in records_with_sr:
        if selected >= limit:
            break
        if not record.get('sr_path') or not os.path.exists(record['sr_path']):
            continue
        try:
            deg = cv2.imread(record['crop_path'], cv2.IMREAD_COLOR)
            sr = cv2.imread(record['sr_path'], cv2.IMREAD_COLOR)
            clean = cv2.imread(
                record.get('clean_crop_path', record['crop_path']),
                cv2.IMREAD_COLOR,
            )
            if deg is None or sr is None or clean is None:
                continue
            h = cfg.HR_HEIGHT
            w = cfg.HR_WIDTH
            deg_r = cv2.resize(deg, (w, h), interpolation=cv2.INTER_CUBIC)
            sr_r = cv2.resize(sr, (w, h), interpolation=cv2.INTER_CUBIC)
            clean_r = cv2.resize(clean, (w, h), interpolation=cv2.INTER_CUBIC)
            row = np.concatenate([deg_r, sr_r, clean_r], axis=1)
            triplets.append(row)
            selected += 1
        except Exception:
            continue

    if triplets:
        grid = np.concatenate(triplets, axis=0)
        cv2.imwrite(sample_path, grid)
        print(f'  Sample grid saved: {sample_path} ({selected} triplets)')


def _process_split(model, records, split_name, cfg):
    """Run TSRN on one split, save SR PNGs, return updated records."""
    out_dir = os.path.join(cfg.SR_CACHE_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    updated = []
    in_range_count = 0
    passthrough_count = 0

    t0 = time.time()

    # Process in chunks
    batch_size = cfg.BATCH_SIZE
    N = len(records)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            chunk = records[start:start + batch_size]
            # Partition chunk into in-range (TSRN) vs passthrough
            in_range_records = []
            in_range_indices = []
            for i, record in enumerate(chunk):
                w = _record_width(record)
                if cfg.MIN_CROP_WIDTH <= w <= cfg.MAX_CROP_WIDTH:
                    in_range_records.append(record)
                    in_range_indices.append(start + i)
                else:
                    # Passthrough — flag and move on
                    new_rec = dict(record)
                    new_rec['sr_path'] = None
                    new_rec['sr_applied'] = False
                    updated.append((start + i, new_rec))
                    passthrough_count += 1

            if in_range_records:
                # Build LR batch and run TSRN
                lr_batch, valid_indices = _build_lr_batch(in_range_records, cfg)
                if lr_batch is None:
                    # All reads failed; mark as passthrough
                    for rec_i, record in enumerate(in_range_records):
                        new_rec = dict(record)
                        new_rec['sr_path'] = None
                        new_rec['sr_applied'] = False
                        updated.append((in_range_indices[rec_i], new_rec))
                        passthrough_count += 1
                    continue

                lr_batch = lr_batch.to(cfg.DEVICE, non_blocking=True)
                out = model(lr_batch)
                sr_batch = out['final_sr']          # (B, 3, 32, 128)

                # Save each SR output and update record
                for local_i, valid_i in enumerate(valid_indices):
                    global_i = in_range_indices[valid_i]
                    record = in_range_records[valid_i]
                    sr_filename = f'{global_i:06d}.png'
                    sr_path = os.path.join(out_dir, sr_filename)
                    _save_sr_png(sr_batch[local_i], sr_path)
                    new_rec = dict(record)
                    new_rec['sr_path'] = sr_path
                    new_rec['sr_applied'] = True
                    updated.append((global_i, new_rec))
                    in_range_count += 1

                # Records whose reads failed
                failed_indices = set(range(len(in_range_records))) - set(valid_indices)
                for failed_i in failed_indices:
                    global_i = in_range_indices[failed_i]
                    record = in_range_records[failed_i]
                    new_rec = dict(record)
                    new_rec['sr_path'] = None
                    new_rec['sr_applied'] = False
                    updated.append((global_i, new_rec))
                    passthrough_count += 1

            if (start // batch_size) % 20 == 0:
                elapsed = time.time() - t0
                rate = (start + batch_size) / max(elapsed, 1e-6)
                eta = (N - start - batch_size) / max(rate, 1e-6)
                print(
                    f'  [{split_name}] {min(start + batch_size, N)} / {N}  '
                    f'({rate:.0f} rec/s, ETA {eta:.0f}s)'
                )

    # Sort updated by original index
    updated.sort(key=lambda pair: pair[0])
    final_records = [rec for _, rec in updated]

    # Save manifest
    manifest_path = os.path.join(out_dir, 'manifest.pkl')
    with open(manifest_path, 'wb') as f:
        pickle.dump(final_records, f)

    elapsed = time.time() - t0
    print(
        f'[{split_name}] {len(final_records)} records in {elapsed:.0f}s — '
        f'{in_range_count} SR, {passthrough_count} passthrough'
    )

    # Save visual samples
    _save_sample_grid(final_records, cfg, split_name, cfg.SAMPLE_DUMP_COUNT)

    return final_records


def build_sr_cache(cfg=None):
    """Phase 2 entry point.

    Returns (sr_train_records, sr_val_records, sr_test_records).
    """
    cfg = cfg or SRCacheConfig()

    print('=' * 70)
    print('Phase 2 — Caching SR outputs for all splits')
    print('=' * 70)
    print(f'TSRN:               {cfg.TSRN_CHECKPOINT_PATH}')
    print(f'TPG:                {cfg.TPG_CHECKPOINT_PATH}')
    print(f'SR cache root:      {cfg.SR_CACHE_ROOT}')
    print(f'Width range:        [{cfg.MIN_CROP_WIDTH}, {cfg.MAX_CROP_WIDTH}]')
    print(f'Batch size:         {cfg.BATCH_SIZE}')
    print(f'Device:             {cfg.DEVICE}')
    print('-' * 70)

    os.makedirs(cfg.SR_CACHE_ROOT, exist_ok=True)
    model = _load_tpgsr_model(cfg)

    sr_train = _process_split(model, final_train_records, 'train', cfg)
    sr_val = _process_split(model, final_val_records, 'val', cfg)
    sr_test = _process_split(model, final_test_records, 'test', cfg)

    print('=' * 70)
    print('Phase 2 complete.')
    print(f'  Train: {len(sr_train)} records')
    print(f'  Val:   {len(sr_val)} records')
    print(f'  Test:  {len(sr_test)} records')
    print(f'  SR cache: {cfg.SR_CACHE_ROOT}')
    print('=' * 70)

    return sr_train, sr_val, sr_test