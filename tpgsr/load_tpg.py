"""
TPG loading helper.

Loads a trained CRNN (v1 or v2) from disk, puts it in eval mode, freezes all
parameters, and returns it ready to be used as the frozen Text Prior Generator
inside TPGSR. Shared by Phase 1, Phase 2, and Phase 3 code.
"""

import torch

from __main__ import ClovaaiCRNN, label_encoder


def load_frozen_tpg(checkpoint_path, cfg, device=None):
    """Instantiate a ClovaaiCRNN matching your baseline config, load weights, freeze.

    Args:
        checkpoint_path: path to the Phase 0 TPG checkpoint (.pth).
        cfg: any Config-like object that exposes IMG_CHANNELS, CNN_OUT_CHANNELS,
            RNN_HIDDEN_SIZE, and DEVICE. The baseline Config (cfg from __main__)
            or your TPGSR config both work.
        device: override device; defaults to cfg.DEVICE.

    Returns:
        Frozen CRNN model in eval mode on the specified device.
    """
    device = device or cfg.DEVICE

    # Use the baseline's IMG_CHANNELS/CNN_OUT_CHANNELS/RNN_HIDDEN_SIZE values.
    # If cfg is the TPGSRConfig (which doesn't have these), fall back to
    # reading them from the baseline Config via __main__.
    from __main__ import cfg as baseline_cfg
    in_ch = getattr(cfg, 'IMG_CHANNELS', baseline_cfg.IMG_CHANNELS)
    cnn_out = getattr(cfg, 'CNN_OUT_CHANNELS', baseline_cfg.CNN_OUT_CHANNELS)
    rnn_hidden = getattr(cfg, 'RNN_HIDDEN_SIZE', baseline_cfg.RNN_HIDDEN_SIZE)

    tpg = ClovaaiCRNN(
        num_classes=label_encoder.num_classes,
        input_channel=in_ch,
        output_channel=cnn_out,
        hidden_size=rnn_hidden,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tpg.load_state_dict(ckpt['model_state_dict'])
    tpg.eval()

    for param in tpg.parameters():
        param.requires_grad = False

    print(
        f"TPG loaded from {checkpoint_path}\n"
        f"  checkpoint epoch: {ckpt['epoch']}, val_acc: {ckpt['val_acc']:.1f}%, "
        f"val_cer: {ckpt['val_cer']:.1f}%"
    )
    return tpg
