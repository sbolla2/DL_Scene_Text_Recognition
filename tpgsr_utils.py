import importlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

DEFAULT_TPGSR_REPO_URL = 'https://github.com/mjq11302010044/TPGSR.git'


def _torch_load_compat(path, **kwargs):
    try:
        return torch.load(path, **kwargs)
    except RuntimeError as error:
        if 'weights_only' not in str(error):
            raise
        return torch.load(path, weights_only=False, **kwargs)


def _normalize_scale_factor(scale_factor):
    normalized = int(round(float(scale_factor)))
    if normalized < 1:
        raise ValueError(f'Invalid TPGSR scale factor: {scale_factor}')
    return normalized


def ensure_tpgsr_repo(repo_dir, repo_url=DEFAULT_TPGSR_REPO_URL):
    repo_path = Path(repo_dir)
    if repo_path.exists():
        return repo_path
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--depth', '1', repo_url, str(repo_path)], check=True)
    return repo_path


def _add_repo_to_path(repo_dir):
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def _extract_generator_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict_G', 'model', 'generator', 'state_dict'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError('Unsupported TPGSR checkpoint format.')
    cleaned = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key[len('module.'):]
        cleaned[new_key] = value
    if not cleaned:
        raise ValueError('No generator weights found in TPGSR checkpoint.')
    return cleaned


def infer_tpgsr_config(state_dict):
    block_numbers = []
    for key in state_dict:
        match = re.match(r'block(\d+)\.', key)
        if match:
            block_numbers.append(int(match.group(1)))
    if not block_numbers:
        raise ValueError('Unable to infer TPGSR block layout from checkpoint.')
    final_block = max(block_numbers)
    upsample_block_ids = {
        int(match.group(1))
        for key in state_dict
        for match in [re.match(rf'block{final_block}\.(\d+)\.conv\.weight$', key)]
        if match
    }
    first_conv = state_dict['block1.0.weight']
    info_tconv1 = state_dict['infoGen.tconv1.weight']
    info_tconv4 = state_dict['infoGen.tconv4.weight']
    return {
        'scale_factor': 2 ** len(upsample_block_ids),
        'srb_nums': final_block - 3,
        'mask': first_conv.shape[1] == 4,
        'hidden_units': first_conv.shape[0] // 2,
        'text_emb': info_tconv1.shape[0],
        'out_text_channels': info_tconv4.shape[1],
    }


def load_tpgsr_generator(repo_dir, checkpoint_path, device, width=256, height=32, use_stn=False, scale_factor=None):
    repo_path = ensure_tpgsr_repo(repo_dir)
    _add_repo_to_path(repo_path)
    tsrn_module = importlib.import_module('model.tsrn')
    checkpoint = _torch_load_compat(checkpoint_path, map_location='cpu')
    state_dict = _extract_generator_state_dict(checkpoint)
    inferred = infer_tpgsr_config(state_dict)
    resolved_scale_factor = _normalize_scale_factor(scale_factor or inferred['scale_factor'])
    generator = tsrn_module.TSRN_TL(
        scale_factor=resolved_scale_factor,
        width=width,
        height=height,
        STN=use_stn,
        srb_nums=inferred['srb_nums'],
        mask=inferred['mask'],
        hidden_units=inferred['hidden_units'],
        text_emb=inferred['text_emb'],
        out_text_channels=inferred['out_text_channels'],
    )
    incompatible = generator.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        allowed_prefixes = () if use_stn else ('tps.', 'stn_head.')
        disallowed = [
            key for key in incompatible.unexpected_keys
            if not any(key.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if disallowed:
            raise ValueError(f'Unexpected TPGSR checkpoint keys: {disallowed}')
    if incompatible.missing_keys:
        allowed_prefixes = ('tps.', 'stn_head.') if not use_stn else ()
        disallowed = [
            key for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if disallowed:
            raise ValueError(f'Missing required TPGSR checkpoint keys: {disallowed}')
    generator.to(device)
    generator.eval()
    return generator, inferred


def _unwrap_model_output(output):
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError('TPGSR model returned an empty output sequence.')
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f'Unsupported TPGSR model output type: {type(output).__name__}')
    return output


def build_text_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    threshold = float(gray.mean())
    return np.where(gray > threshold, 0.0, 1.0).astype(np.float32)


def downscale_for_tpgsr(image_bgr, scale_factor):
    scale_factor = _normalize_scale_factor(scale_factor)
    height, width = image_bgr.shape[:2]
    lr_width = max(1, math.ceil(width / float(scale_factor)))
    lr_height = max(1, math.ceil(height / float(scale_factor)))
    return cv2.resize(image_bgr, (lr_width, lr_height), interpolation=cv2.INTER_CUBIC)


def image_to_tpgsr_tensor(image_bgr, use_mask):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    channels = [np.transpose(image_rgb, (2, 0, 1))]
    if use_mask:
        channels.append(build_text_mask(image_bgr)[None, ...])
    array = np.concatenate(channels, axis=0)
    return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)


def _cosine_ramp_1d(n, rising, device, dtype):
    """Raised-cosine ramp of length n in (0, 1); rising=True goes small→1, else 1→small."""
    if n <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    if n == 1:
        return torch.ones(1, device=device, dtype=dtype)
    t = torch.linspace(0.0, math.pi, n, device=device, dtype=dtype)
    if rising:
        ramp = 0.5 - 0.5 * torch.cos(t)
    else:
        ramp = 0.5 + 0.5 * torch.cos(t)
    return ramp.clamp_min_(1e-6)


def _tile_blend_weights_2d(tile_out_h, tile_out_w, scale_factor, top_lr, left_lr, bottom_lr, right_lr,
                           canvas_h_lr, canvas_w_lr, overlap_lr, device, dtype):
    """
    Per-tile multiplicative weights (cosine ramps) on sides that border another LR tile.
    Canvas edges stay at 1 so borders are not dimmed. Overlap depth follows tile_overlap in LR.
    """
    overlap_lr = max(0, int(overlap_lr))
    if overlap_lr == 0:
        return torch.ones(tile_out_h, tile_out_w, device=device, dtype=dtype)
    ov = min(overlap_lr * scale_factor, tile_out_h, tile_out_w)
    if ov <= 0:
        return torch.ones(tile_out_h, tile_out_w, device=device, dtype=dtype)
    # Keep top/bottom (or left/right) ramps from overlapping on very small HR tiles.
    ov_h = min(ov, tile_out_h // 2) if (top_lr > 0 and bottom_lr < canvas_h_lr) else min(ov, tile_out_h)
    ov_w = min(ov, tile_out_w // 2) if (left_lr > 0 and right_lr < canvas_w_lr) else min(ov, tile_out_w)
    wv = torch.ones(tile_out_h, device=device, dtype=dtype)
    wh = torch.ones(tile_out_w, device=device, dtype=dtype)
    if top_lr > 0:
        wv[:ov_h] *= _cosine_ramp_1d(ov_h, True, device, dtype)
    if bottom_lr < canvas_h_lr:
        wv[-ov_h:] *= _cosine_ramp_1d(ov_h, False, device, dtype)
    if left_lr > 0:
        wh[:ov_w] *= _cosine_ramp_1d(ov_w, True, device, dtype)
    if right_lr < canvas_w_lr:
        wh[-ov_w:] *= _cosine_ramp_1d(ov_w, False, device, dtype)
    return wv[:, None] * wh[None, :]


def _run_tiled(model, lr_tensor, scale_factor, tile_size=None, tile_overlap=32):
    scale_factor = _normalize_scale_factor(scale_factor)
    _, _, height, width = lr_tensor.shape
    if tile_size is None or (height <= tile_size and width <= tile_size):
        return _unwrap_model_output(model(lr_tensor))
    tile_size = int(tile_size)
    if tile_size < 1:
        return _unwrap_model_output(model(lr_tensor))
    # LR overlap must be < tile_size so stride = tile_size - overlap is >= 1 (avoids max(1, neg)=1 blowup).
    overlap_lr = max(0, min(int(tile_overlap), tile_size - 1))
    stride = tile_size - overlap_lr
    output = None
    weight = None
    for top in range(0, height, stride):
        bottom = min(top + tile_size, height)
        top = max(0, bottom - tile_size)
        for left in range(0, width, stride):
            right = min(left + tile_size, width)
            left = max(0, right - tile_size)
            tile = lr_tensor[:, :, top:bottom, left:right]
            tile_output = _unwrap_model_output(model(tile))
            if output is None:
                output = torch.zeros(
                    (tile_output.shape[0], tile_output.shape[1], height * scale_factor, width * scale_factor),
                    dtype=tile_output.dtype,
                    device=tile_output.device,
                )
                weight = torch.zeros_like(output)
            out_top = top * scale_factor
            out_left = left * scale_factor
            out_bottom = out_top + tile_output.shape[-2]
            out_right = out_left + tile_output.shape[-1]
            blend = _tile_blend_weights_2d(
                tile_output.shape[-2],
                tile_output.shape[-1],
                scale_factor,
                top,
                left,
                bottom,
                right,
                height,
                width,
                overlap_lr,
                tile_output.device,
                tile_output.dtype,
            )
            blend = blend.view(1, 1, blend.shape[0], blend.shape[1])
            weighted = tile_output * blend
            output[:, :, out_top:out_bottom, out_left:out_right] += weighted
            weight[:, :, out_top:out_bottom, out_left:out_right] += blend
    return output / weight.clamp_min_(1e-6)


@torch.inference_mode()
def restore_image_with_tpgsr(model, image_bgr, device, scale_factor, use_mask=True, tile_size=512, tile_overlap=32):
    scale_factor = _normalize_scale_factor(scale_factor)
    original_height, original_width = image_bgr.shape[:2]
    degraded_lr = downscale_for_tpgsr(image_bgr, scale_factor)
    lr_tensor = image_to_tpgsr_tensor(degraded_lr, use_mask=use_mask).to(device)
    restored = _run_tiled(model, lr_tensor, scale_factor=scale_factor, tile_size=tile_size, tile_overlap=tile_overlap)
    restored = restored[:, :3].clamp_(0.0, 1.0)
    restored_np = restored[0].detach().cpu().permute(1, 2, 0).numpy()
    restored_rgb = (restored_np * 255.0).round().astype(np.uint8)
    restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
    if restored_bgr.shape[0] < original_height or restored_bgr.shape[1] < original_width:
        restored_bgr = cv2.resize(restored_bgr, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    else:
        restored_bgr = restored_bgr[:original_height, :original_width]
        if restored_bgr.shape[0] != original_height or restored_bgr.shape[1] != original_width:
            restored_bgr = cv2.resize(restored_bgr, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    return restored_bgr