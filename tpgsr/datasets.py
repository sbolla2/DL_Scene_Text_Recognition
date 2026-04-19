"""
Phase 0 — clean-crop dataset for TPG-CRNN training.

Thin variant of the baseline TotalTextDataset that loads the clean pre-degradation
crop (record['clean_crop_path']) instead of the degraded one (record['crop_path']).

Everything else — grayscale conversion, keep-ratio resize with replicate padding,
normalization, label encoding — is reused from the baseline via inheritance.
"""

import cv2
import torch
from PIL import Image

from __main__ import TotalTextDataset


class CleanCropDataset(TotalTextDataset):
    """Same preprocessing as TotalTextDataset but reads clean crops.

    Falls back to record['crop_path'] if clean_crop_path is missing, which
    happens for non-benchmark records that never went through the degradation
    cache. In practice all records in final_{train,val,test}_records have
    clean_crop_path set.
    """

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = record.get('clean_crop_path', record['crop_path'])
        text = record['text']

        crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if crop is None:
            return None

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = self._resize_keep_ratio_pad(crop)

        pil_img = Image.fromarray(crop)
        if self.pil_transform:
            pil_img = self.pil_transform(pil_img)
        img_t = self.to_tensor(pil_img)
        label = self.label_encoder.encode(text)
        return img_t, label, len(label)
