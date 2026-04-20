"""
Phase 3 dataset — TRBA input reader for SR-cached records.

Hybrid read logic:
  - If sr_applied=True: read sr_path (3-ch SR PNG), convert to grayscale
  - Else: read crop_path (degraded), convert to grayscale

Applies TRBA's keep-ratio-resize-pad logic.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SRTRBADataset(Dataset):
    def __init__(self, records, label_encoder, cfg, augment=False):
        self.records = records
        self.label_encoder = label_encoder
        self.cfg = cfg
        self.augment = augment
        self.max_width = cfg.IMG_WIDTH
        self.height = cfg.IMG_HEIGHT

        self.pil_transform = None
        if augment:
            aug = getattr(cfg, 'AUGMENTATION', None)
            if aug:
                self.pil_transform = transforms.Compose([
                    transforms.RandomAffine(
                        degrees=aug.get('affine_degrees', 2.0),
                        shear=aug.get('affine_shear', 4.0),
                    ),
                    transforms.RandomApply([
                        transforms.ColorJitter(
                            brightness=aug.get('brightness', 0.15),
                            contrast=aug.get('contrast', 0.15),
                        )
                    ], p=aug.get('color_prob', 0.30)),
                ])

        channel_mean = [0.5] * cfg.IMG_CHANNELS
        channel_std = [0.5] * cfg.IMG_CHANNELS
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=channel_mean, std=channel_std),
        ])

    def __len__(self):
        return len(self.records)

    def _resize_keep_ratio_pad(self, crop):
        if crop.ndim == 2:
            h, w = crop.shape
        else:
            h, w = crop.shape[:2]

        if h != self.height:
            new_w = max(1, int(round(w * (self.height / max(h, 1)))))
            crop = cv2.resize(crop, (new_w, self.height), interpolation=cv2.INTER_CUBIC)
            h, w = crop.shape[:2]

        target_w = min(self.max_width, max(1, int(np.ceil(self.height * (w / max(h, 1))))))
        if w != target_w:
            crop = cv2.resize(crop, (target_w, self.height), interpolation=cv2.INTER_CUBIC)

        if target_w < self.max_width:
            pad_w = self.max_width - target_w
            border_value = 0 if crop.ndim == 2 else [0] * crop.shape[2]
            crop = cv2.copyMakeBorder(
                crop, 0, 0, 0, pad_w, cv2.BORDER_REPLICATE, value=border_value
            )

        return crop[:, :self.max_width]

    def __getitem__(self, idx):
        record = self.records[idx]

        if record.get('sr_applied') and record.get('sr_path'):
            img_path = record['sr_path']
        else:
            img_path = record['crop_path']

        crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if crop is None:
            return None

        if self.cfg.IMG_CHANNELS == 1:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop = self._resize_keep_ratio_pad(crop)

        pil_img = Image.fromarray(crop)
        if self.pil_transform:
            pil_img = self.pil_transform(pil_img)
        img_t = self.to_tensor(pil_img)

        text = record['text']
        label = self.label_encoder.encode(text)
        return img_t, label, len(label)
