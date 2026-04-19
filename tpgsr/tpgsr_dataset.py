"""
TPGSR Dataset and collate function.

Produces paired (LR, HR, mask, text) batches for TSRN training:
  - LR: degraded crop, resized to (16, 64) RGB
  - HR: clean crop, resized to (32, 128) RGB — the ground truth
  - mask: polygon mask, resized to (16, 64) binary, added as 4th LR channel
  - text: GT label (encoded for CTC)

Only uses records where the ORIGINAL clean crop width is in
[cfg.MIN_TRAIN_WIDTH, cfg.MAX_TRAIN_WIDTH]. Crops outside this range will
bypass TSRN at inference (Phase 2 hybrid policy).

Reads image bytes from disk on the fly — same pattern as your existing
TotalTextDataset.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_tensor_01(img_rgb_uint8):
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
    return t


def _to_tensor_tanh(img_rgb_uint8):
    """(H, W, 3) uint8 -> (3, H, W) float32 in [-1, 1] (TSRN's input/output range)."""
    return _to_tensor_01(img_rgb_uint8) * 2.0 - 1.0


def _to_mask_tensor(mask_uint8, target_size_hw):
    """(H, W) uint8 -> (1, target_H, target_W) float32 binarized to {-1, +1}.

    We use [-1, +1] for the mask channel so it matches the RGB tanh range.
    """
    target_h, target_w = target_size_hw
    if mask_uint8.ndim == 3:
        mask_uint8 = mask_uint8[..., 0]
    resized = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    binarized = (resized > 127).astype(np.float32)
    t = torch.from_numpy(binarized).unsqueeze(0)                 # (1, H, W)
    return t * 2.0 - 1.0                                          # -> [-1, +1]


def filter_records_by_width(records, min_w, max_w):
    """Keep records whose clean crop has width in [min_w, max_w]."""
    kept = []
    for rec in records:
        path = rec.get('clean_crop_path', rec['crop_path'])
        # Use bbox from record if available (faster than reading the image).
        bbox = rec.get('crop_bbox')
        if bbox:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            w = img.shape[1]
        if min_w <= w <= max_w:
            kept.append(rec)
    return kept


class TPGSRDataset(Dataset):
    """LR/HR paired dataset for TPGSR training."""

    def __init__(self, records, label_encoder, cfg):
        """
        Args:
            records: list of dicts from your benchmark cache. Each dict must have:
                - 'crop_path' (degraded)
                - 'clean_crop_path' (clean HR source)
                - 'mask_path' (polygon mask)
                - 'text' (already normalized to charset)
            label_encoder: your LabelEncoder (same as baseline).
            cfg: TPGSRConfig instance.
        """
        self.records = records
        self.label_encoder = label_encoder
        self.cfg = cfg

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        lr_path = rec['crop_path']
        hr_path = rec.get('clean_crop_path', rec['crop_path'])
        mask_path = rec.get('mask_path')
        text = rec['text']

        lr_bgr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_bgr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        if lr_bgr is None or hr_bgr is None:
            return None

        # BGR -> RGB
        lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        hr_rgb = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)

        # Force-resize to canonical dimensions
        lr_resized = cv2.resize(
            lr_rgb, (self.cfg.LR_WIDTH, self.cfg.LR_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )
        hr_resized = cv2.resize(
            hr_rgb, (self.cfg.HR_WIDTH, self.cfg.HR_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

        lr_tensor_rgb = _to_tensor_tanh(lr_resized)          # (3, 16, 64) in [-1, 1]
        hr_tensor = _to_tensor_tanh(hr_resized)               # (3, 32, 128) in [-1, 1]

        # Mask: always load/synthesize a (1, 16, 64) channel for the 4th input slot
        if mask_path:
            mask_uint8 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_uint8 is None:
                mask_uint8 = np.full(
                    (self.cfg.LR_HEIGHT, self.cfg.LR_WIDTH), 255, dtype=np.uint8
                )
        else:
            mask_uint8 = np.full(
                (self.cfg.LR_HEIGHT, self.cfg.LR_WIDTH), 255, dtype=np.uint8
            )
        mask_tensor = _to_mask_tensor(
            mask_uint8, (self.cfg.LR_HEIGHT, self.cfg.LR_WIDTH)
        )                                                      # (1, 16, 64) in [-1, +1]

        # Concatenate mask as 4th channel of LR input
        lr_tensor = torch.cat([lr_tensor_rgb, mask_tensor], dim=0)   # (4, 16, 64)

        # Label (for CTC: concatenated per-batch, lengths returned separately)
        label = self.label_encoder.encode(text)               # 1-D LongTensor
        label_length = len(label)

        return lr_tensor, hr_tensor, label, label_length, text


def tpgsr_collate_fn(batch):
    """Drop None items (bad reads), stack tensors, concat labels with lengths."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    lrs, hrs, labels, lengths, texts = zip(*batch)
    lr_batch = torch.stack(lrs, 0)
    hr_batch = torch.stack(hrs, 0)
    label_tensor = torch.cat(labels, 0)
    length_tensor = torch.LongTensor(lengths)
    return lr_batch, hr_batch, label_tensor, length_tensor, list(texts)
