"""
Phase 3 dataset — CRNN input reader for SR-cached records.

Hybrid read logic:
  - If sr_applied=True: read sr_path (3-ch SR PNG), convert to grayscale
  - Else: read crop_path (original degraded), convert to grayscale

Then applies the same resize-keep-ratio-pad logic the baseline CRNN uses.
"""

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SRCRNNDataset(Dataset):
    """Reads SR (when applied) or degraded (passthrough) as grayscale for CRNN."""

    def __init__(self, records, label_encoder, cfg, augment=False):
        self.records = records
        self.label_encoder = label_encoder
        self.cfg = cfg
        self.augment = augment
        self.max_width = cfg.IMG_WIDTH
        self.height = cfg.IMG_HEIGHT
        # Match baseline CRNN preprocessing
        self.pil_transform = None
        if augment:
            aug = getattr(cfg, 'AUGMENTATION', None)
            if aug:
                self.pil_transform = transforms.Compose([
                    transforms.RandomAffine(
                        degrees=aug['affine_degrees'],
                        shear=aug['affine_shear'],
                    ),
                    transforms.RandomApply([
                        transforms.ColorJitter(
                            brightness=aug['brightness'],
                            contrast=aug['contrast'],
                        )
                    ], p=aug.get('color_prob', 0.35)),
                ])
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.records)

    def _resize_keep_ratio_pad(self, crop):
        """Same logic as TotalTextDataset for consistency with baseline CRNN."""
        h, w = crop.shape
        if h != self.height:
            new_w = max(1, int(round(w * (self.height / max(h, 1)))))
            crop = cv2.resize(crop, (new_w, self.height), interpolation=cv2.INTER_CUBIC)
            _, w = crop.shape

        if w > self.max_width:
            crop = cv2.resize(crop, (self.max_width, self.height), interpolation=cv2.INTER_CUBIC)
            w = self.max_width

        if w < self.max_width:
            pad_w = self.max_width - w
            crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad_w, cv2.BORDER_REPLICATE)

        return crop[:, :self.max_width]

    def __getitem__(self, idx):
        record = self.records[idx]

        # Choose source: SR if applied, else degraded
        if record.get('sr_applied') and record.get('sr_path'):
            img_path = record['sr_path']
        else:
            img_path = record['crop_path']

        crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if crop is None:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = self._resize_keep_ratio_pad(crop)

        pil_img = Image.fromarray(crop)
        if self.pil_transform is not None:
            pil_img = self.pil_transform(pil_img)
        img_t = self.to_tensor(pil_img)

        text = record['text']
        label = self.label_encoder.encode(text)
        return img_t, label, len(label)
