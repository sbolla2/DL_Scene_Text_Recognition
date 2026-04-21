"""
SVTRv2 Phase 2 — build SR cache using TRBA-TPGSR and write OpenOCR-format label files.

Since Option 4 is the hybrid approach (SVTRv2 consuming SR outputs from TRBA-guided
TSRN), this script:
  1. Loads the trained TRBA-TPGSR model (TSRN + frozen TRBA-TPG)
  2. Runs it on all benchmark crops (train/val/test)
  3. Saves SR outputs as 32x128 RGB PNGs (SVTRv2's expected input format)
  4. Writes OpenOCR-format label files pointing at SR paths (for SR crops) or
     degraded paths (for passthrough crops outside [40, 160] width)

Reuses the existing `build_sr_cache_trba` logic but adds SVTRv2 label-file writing.

Two modes:
  A) SR cache already exists -> only write SVTRv2 label files
  B) SR cache missing -> call build_sr_cache_trba(), then write label files
"""

import os
import pickle
from pathlib import Path


def _sr_cache_exists(sr_root):
    required = [
        f'{sr_root}/train/manifest.pkl',
        f'{sr_root}/val/manifest.pkl',
        f'{sr_root}/test/manifest.pkl',
    ]
    return all(os.path.exists(p) for p in required)


def _load_sr_manifests(sr_root):
    with open(f'{sr_root}/train/manifest.pkl', 'rb') as f:
        train = pickle.load(f)
    with open(f'{sr_root}/val/manifest.pkl', 'rb') as f:
        val = pickle.load(f)
    with open(f'{sr_root}/test/manifest.pkl', 'rb') as f:
        test = pickle.load(f)
    return train, val, test


def _normalize_svtrv2_text(text, charset):
    import unicodedata
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('ascii')
    text = ''.join(ch for ch in text if ch in charset)
    return text


def _filter_records_for_svtrv2(records, max_text_length, charset):
    kept = []
    skipped = {'empty': 0, 'null_label': 0, 'too_long': 0, 'missing_file': 0}

    for record in records:
        normalized = _normalize_svtrv2_text(record['text'], charset)
        if not normalized:
            skipped['empty'] += 1
            continue
        if normalized in {'.', '#'}:
            skipped['null_label'] += 1
            continue
        if len(normalized) > max_text_length:
            skipped['too_long'] += 1
            continue

        # Resolve the actual image path we will feed to SVTRv2:
        # SR crops -> sr_path, passthrough -> crop_path (degraded)
        if record.get('sr_applied') and record.get('sr_path'):
            image_path = record['sr_path']
        else:
            image_path = record['crop_path']

        if not os.path.exists(image_path):
            skipped['missing_file'] += 1
            continue

        updated = dict(record)
        updated['text'] = normalized
        updated['svtrv2_image_path'] = image_path
        kept.append(updated)

    return kept, skipped


def _write_svtrv2_sr_label_file(records, label_path, data_root):
    """Write OpenOCR-style tab-separated label file: <rel_path>\\t<text>"""
    data_root = Path(data_root).resolve()
    n_written = 0
    n_outside_root = 0

    with open(label_path, 'w', encoding='utf-8') as handle:
        for record in records:
            image_path = Path(record['svtrv2_image_path']).resolve()
            try:
                rel_path = image_path.relative_to(data_root)
            except ValueError:
                # If SR cache lives outside data root, use absolute path
                rel_path = image_path
                n_outside_root += 1
            handle.write(f'{rel_path.as_posix()}\t{record["text"]}\n')
            n_written += 1

    return n_written, n_outside_root


def build_svtrv2_sr_cache(
    sr_root,
    label_dir,
    data_root,
    max_text_length=25,
    charset=None,
    rebuild_sr=False,
):
    """Build SVTRv2 SR label files pointing at an already-generated TRBA SR cache.

    If sr_root's manifests don't exist and rebuild_sr=True, calls build_sr_cache_trba.

    Args:
        sr_root: path to TRBA SR cache, e.g. '/content/data/sr_cache_trba/heavy'
        label_dir: where to write the new label files
        data_root: the root path that OpenOCR label paths are relative to
                   (should be the same SVTRV2_DATA_ROOT used in baseline training)
        max_text_length: matches SVTRV2_MAX_TEXT_LENGTH
        charset: the 94-char ASCII charset (SVTRV2_BENCHMARK_CHARSET)
        rebuild_sr: if True and sr_root is empty, call build_sr_cache_trba()

    Returns:
        dict with paths to the 3 written label files and records.
    """
    if charset is None:
        charset = ''.join(chr(code) for code in range(33, 127))

    # Rebuild TRBA SR cache if needed
    if not _sr_cache_exists(sr_root):
        if not rebuild_sr:
            raise FileNotFoundError(
                f'TRBA SR cache not found at {sr_root}. '
                f'Set rebuild_sr=True to regenerate, or run build_sr_cache_trba() first.'
            )
        print(f'SR cache missing at {sr_root}, regenerating...')
        from tpgsr.build_sr_cache_trba import build_sr_cache_trba
        build_sr_cache_trba()

    # Load manifests
    train_raw, val_raw, test_raw = _load_sr_manifests(sr_root)

    n_sr_train = sum(1 for r in train_raw if r.get('sr_applied'))
    n_sr_val = sum(1 for r in val_raw if r.get('sr_applied'))
    n_sr_test = sum(1 for r in test_raw if r.get('sr_applied'))
    print(
        f'TRBA SR cache loaded from {sr_root}:\n'
        f'  train: {len(train_raw)} records ({n_sr_train} SR / {len(train_raw) - n_sr_train} passthrough)\n'
        f'  val:   {len(val_raw)} records ({n_sr_val} SR / {len(val_raw) - n_sr_val} passthrough)\n'
        f'  test:  {len(test_raw)} records ({n_sr_test} SR / {len(test_raw) - n_sr_test} passthrough)'
    )

    # Filter per SVTRv2 rules (charset + length + file existence)
    train_filt, skip_train = _filter_records_for_svtrv2(train_raw, max_text_length, charset)
    val_filt, skip_val = _filter_records_for_svtrv2(val_raw, max_text_length, charset)
    test_filt, skip_test = _filter_records_for_svtrv2(test_raw, max_text_length, charset)

    print(f'\nFiltered for SVTRv2:')
    print(f'  train: {len(train_filt)} kept (skipped: {skip_train})')
    print(f'  val:   {len(val_filt)} kept (skipped: {skip_val})')
    print(f'  test:  {len(test_filt)} kept (skipped: {skip_test})')

    # Write label files
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    train_label = label_dir / 'svtrv2_sr_train.txt'
    val_label = label_dir / 'svtrv2_sr_val.txt'
    test_label = label_dir / 'svtrv2_sr_test.txt'

    n_train, out_train = _write_svtrv2_sr_label_file(train_filt, train_label, data_root)
    n_val, out_val = _write_svtrv2_sr_label_file(val_filt, val_label, data_root)
    n_test, out_test = _write_svtrv2_sr_label_file(test_filt, test_label, data_root)

    print(f'\nLabel files written:')
    print(f'  {train_label}: {n_train} entries ({out_train} outside data_root)')
    print(f'  {val_label}:   {n_val} entries ({out_val} outside data_root)')
    print(f'  {test_label}:  {n_test} entries ({out_test} outside data_root)')

    if out_train + out_val + out_test > 0:
        print(
            'WARNING: Some image paths fall outside SVTRV2_DATA_ROOT and were written as absolute paths. '
            'OpenOCR may have trouble resolving these — consider copying the SR cache into the data_root tree '
            'or overriding SVTRV2_DATA_ROOT to a parent directory covering both.'
        )

    return {
        'train_label_path': train_label,
        'val_label_path': val_label,
        'test_label_path': test_label,
        'train_records': train_filt,
        'val_records': val_filt,
        'test_records': test_filt,
    }
