# Enhancing Degraded Scene Text Recognition through Text Prior Guided Super-Resolution

## Sahit Bolla and Sahil Chute

This repository contains notebook-first experiments for scene text recognition under heavy degradation, centered on a text-prior-guided super-resolution pipeline referred to in the code as TPGSR.

The project combines three recognizer families:

- CRNN: VGG + BiLSTM + CTC recognizer used both as a baseline and as a frozen text prior.
- TRBA: TPS-ResNet-BiLSTM-Attn recognizer adapted for the same multi-phase TPGSR workflow.
- SVTRv2: OpenOCR-based recognizer fine-tuned on the shared crop pipeline and, optionally, TRBA-guided SR outputs.

### Demo notebook (no edits required)

If you just want to see the pipeline end-to-end without changing any code, open the demo notebook and run it top to bottom.

1. Open [STR_Demo.ipynb](STR_Demo.ipynb) in Jupyter or Colab.
2. Run all cells in order. No edits or parameter changes are needed.

## Repository layout

```text
.
├── STR_AllModels_Clean.ipynb
├── STR_Demo.ipynb
├── CRNN/
│   ├── TPGSR_CRNN_HeavyDegradation.ipynb
│   └── tpgsr/
├── TRBA/
│   ├── TPGSR_TRBA_HeavyDegradation.ipynb
│   └── tpgsr_trba/
└── SVTR/
    ├── TPGSR_SVTRv2_HeavyDegradation.ipynb
    └── tpgsr/
```

## What the code implements

The codebase is organized around a staged pipeline.

### Phase 0: Train a clean text prior generator

- CRNN path: [CRNN/tpgsr/train_tpg.py](CRNN/tpgsr/train_tpg.py)
- TRBA path: [TRBA/tpgsr_trba/train_trba_tpg.py](TRBA/tpgsr_trba/train_trba_tpg.py)

The recognizer is trained on clean cropped word images and saved as the text prior used in later SR training.

### Phase 1: Train TPGSR

- CRNN-guided TPGSR: [CRNN/tpgsr/train_tpgsr.py](CRNN/tpgsr/train_tpgsr.py)
- TRBA-guided TPGSR: [TRBA/tpgsr_trba/train_tpgsr_trba.py](TRBA/tpgsr_trba/train_tpgsr_trba.py)

This stage trains a TSRN-based super-resolution model with a frozen recognizer acting as the text prior. The SR model consumes RGB crops plus a polygon mask and optimizes a joint pixel and text-consistency objective.

### Phase 2: Build an SR cache

- CRNN cache builder: [CRNN/tpgsr/build_sr_cache.py](CRNN/tpgsr/build_sr_cache.py)
- TRBA cache builder: [TRBA/tpgsr/build_sr_cache_trba.py](TRBA/tpgsr/build_sr_cache_trba.py)
- SVTRv2 label-file bridge: [SVTR/tpgsr/build_svtrv2_sr_cache.py](SVTR/tpgsr/build_svtrv2_sr_cache.py)

This stage runs the trained SR model over cropped text instances, saves SR outputs, and records whether each crop was actually super-resolved or passed through unchanged. A width filter is used throughout the codebase so very narrow or very wide crops bypass SR instead of being forced through a distribution the model was not trained on.

### Phase 3: Fine-tune the recognizer on SR outputs

- CRNN fine-tuning: [CRNN/tpgsr/finetune_crnn.py](CRNN/tpgsr/finetune_crnn.py)
- TRBA fine-tuning: [TRBA/tpgsr_trba/finetune_trba.py](TRBA/tpgsr_trba/finetune_trba.py)

The recognizer is adapted from the degraded-input distribution to the SR-output distribution using the manifests from Phase 2.

### Phase 4: Evaluate the end-to-end pipeline

- CRNN evaluation: [CRNN/tpgsr/evaluate_tpgsr.py](CRNN/tpgsr/evaluate_tpgsr.py)
- TRBA evaluation: [TRBA/tpgsr_trba/evaluate_trba_tpgsr.py](TRBA/tpgsr_trba/evaluate_trba_tpgsr.py)

The evaluation scripts reload the fine-tuned recognizer, run on the SR cache test manifest, and report word accuracy and character error rate.

## Main entry points

### Notebooks

- [STR_AllModels_Clean.ipynb](STR_AllModels_Clean.ipynb): This is a reference (not baseline) notebook that downloads data, builds clean crop caches, trains/evaluates CRNN and TRBA flows, and includes an SVTRv2 section built on OpenOCR. It shows how STR models work on clean scene text image inputs as intended.
- [CRNN/TPGSR_CRNN_HeavyDegradation.ipynb](CRNN/TPGSR_CRNN_HeavyDegradation.ipynb): focused TPGSR-aided CRNN heavy-degradation workflow built on top of the reference notebook.
- [TRBA/TPGSR_TRBA_HeavyDegradation.ipynb](TRBA/TPGSR_TRBA_HeavyDegradation.ipynb): focused TPGSR-aided TRBA heavy-degradation workflow built on top of the reference notebook.
- [SVTR/TPGSR_SVTRv2_HeavyDegradation.ipynb](SVTR/TPGSR_SVTRv2_HeavyDegradation.ipynb): focused TPGSR-aided SVTRv2 heavy-degradation workflow built on top of the reference notebook.

### Python modules

The Python packages under [CRNN/tpgsr](CRNN/tpgsr), [TRBA/tpgsr_trba](TRBA/tpgsr_trba), and [SVTR/tpgsr](SVTR/tpgsr) are not standalone CLIs. They are designed to be imported from notebook sessions after earlier cells have already created globals such as `final_train_records`, `label_encoder`, model builders, and baseline configs in `__main__`.

That coupling is visible directly in the module imports. For example, [CRNN/tpgsr/train_tpg.py](CRNN/tpgsr/train_tpg.py) and [CRNN/tpgsr/finetune_crnn.py](CRNN/tpgsr/finetune_crnn.py) import datasets, encoders, and model constructors from `__main__`, so notebook execution order matters.

## Data pipeline

The repository works with cropped word instances derived from scene-text datasets.

- TextOCR is downloaded via `kagglehub` and sampled by image before crop extraction in the main notebook.
- Total-Text is also downloaded via `kagglehub`, then reshaped into `data/totaltext/images/{train,test}` and `data/totaltext/annotations`.
- Crops are extracted from polygons or bounding boxes, and the crop metadata is carried forward as Python dictionaries.

The main notebook builds reusable crop records and then derives model-specific normalized views:

- CRNN uses a lowercase alphanumeric charset.
- TRBA uses an attention-style label converter and its own charset settings.
- SVTRv2 normalizes labels into a benchmark-style printable ASCII charset and writes OpenOCR label files.

The SR cache manifests used later in the pipeline carry fields such as the original crop path, optional SR path, label text, and whether SR was applied.

## Configuration

Configuration is implemented as Python classes rather than YAML for the CRNN and TRBA flows.

- Phase 0 CRNN config: [CRNN/tpgsr/tpg_config.py](CRNN/tpgsr/tpg_config.py)
- Phase 1 CRNN config: [CRNN/tpgsr/tpgsr_config.py](CRNN/tpgsr/tpgsr_config.py)
- Phase 2 CRNN cache config: [CRNN/tpgsr/sr_cache_config.py](CRNN/tpgsr/sr_cache_config.py)
- Phase 3 CRNN fine-tune config: [CRNN/tpgsr/finetune_config.py](CRNN/tpgsr/finetune_config.py)
- Phase 0 TRBA config: [TRBA/tpgsr_trba/trba_tpg_config.py](TRBA/tpgsr_trba/trba_tpg_config.py)
- Phase 1 TRBA config: [TRBA/tpgsr_trba/tpgsr_trba_config.py](TRBA/tpgsr_trba/tpgsr_trba_config.py)
- Phase 2 TRBA cache config: [TRBA/tpgsr_trba/sr_cache_trba_config.py](TRBA/tpgsr_trba/sr_cache_trba_config.py)
- Phase 3 TRBA fine-tune config: [TRBA/tpgsr_trba/finetune_trba_config.py](TRBA/tpgsr_trba/finetune_trba_config.py)

Important defaults visible in the code:

- TPGSR trains a 2x SR model from 16x64 low-resolution inputs to 32x128 outputs.
- The SR model uses 4-channel inputs: RGB crop plus binary polygon mask.
- CRNN-guided TPGSR uses a CTC-based text loss.
- TRBA-guided TPGSR uses a cross-entropy text loss.
- Width filtering is used in both training and inference to avoid applying SR outside the supported crop-width range.

SVTRv2 is handled differently. The notebook section in [STR_AllModels_Clean.ipynb](STR_AllModels_Clean.ipynb) clones OpenOCR on demand, installs its requirements, downloads pretrained checkpoints, writes a custom character dictionary, generates OpenOCR config files, and launches fine-tuning from inside the notebook.

## Dependencies

Core packages used directly by the repository:

- `torch`, `torchvision`, `torchaudio`
- `opencv-python-headless`
- `Pillow`
- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `editdistance`
- `tqdm`
- `kagglehub`
- `PyYAML`
- `gdown`

External code and pretrained model ecosystems used by the notebooks:

- ClovaAI-style CRNN/TRBA checkpoints and model conventions.
- OpenOCR for the SVTRv2 branch, cloned dynamically by the notebook.

For a local environment, start with a Python 3.10+ virtual environment and install the packages above. If you plan to run the SVTRv2 branch, the notebook will additionally install OpenOCR's own requirements.

## Running the project

### Recommended path

Use the notebooks, not the Python modules directly.

1. Import to Google Drive for Colab: [STR_AllModels_Clean.ipynb](STR_AllModels_Clean.ipynb) for the reference integrated workflow on clean inputs, or pick one of the model-specific TPGSR-aided and degradation-applied notebooks under [CRNN](CRNN), [TRBA](TRBA), or [SVTR](SVTR).
2. Run the first cell first to mount the notebook to mount Google Drive in Colab.
3. If running a model-specific notebook, create the /tpgsr and /weights directories at /content/ path and add the corresponding files to it from this repo.
4. Run the notebook.
