# CoughClassificationProject

Audio-based classification of elderly coughs (infectious vs. non-infectious) using PyTorch neural networks.

## Project Structure

```text
CoughClassificationProject/
│
├── main.py                                 # Primary entrypoint
├── pyproject.toml                          # Project metadata, dependencies (uv)
│
├── core/                                   # Framework library
│   │
│   ├── experiment.py                       # Top-level workflow
│   ├── experiment_config.py
│   │
│   ├── data_pipeline/                      # Audio → feature-extraction pipeline
│   │   │
│   │   ├── abstract.py                     #   ABCs: SourceReader, Segmenter, Transformer, Padder, Splitter
│   │   ├── pipeline.py
│   │   ├── intermediary.py                 #   Core types: SourceSeries, Example, DevelopmentFold, DataSplit
│   │   ├── dataset.py
│   │   ├── data_pipeline_config.py
│   │   │
│   │   ├── preprocessing/                  #   Concrete transforms
│   │   │   │
│   │   │   ├── segmentation.py
│   │   │   ├── transform.py                #     MFCC, LogMelSpectrogram
│   │   │   ├── padding.py
│   │   │   └── sampling.py
│   │   │
│   │   ├── source_reader/                  #   Dataset-specific I/O
│   │   │   │
│   │   │   └── elderly_cough_audio/        #     Audio read, cache, resample, metadata translation
│   │   │       │
│   │   │       ├── source_reader.py        #       ElderlyCoughAudioSourceReader
│   │   │       ├── audio_reader.py         #       WAV loading
│   │   │       ├── audio_cache.py          #       Processed-audio caching
│   │   │       ├── audio_resampler.py      #       Sample-rate normalization
│   │   │       ├── metadata_reader.py      #       CSV metadata parsing
│   │   │       └── metadata_translator.py  #       Label mapping
│   │   │
│   │   └── stratifier/                     #   Patient-grouped train/test splitting
│   │       │
│   │       ├── data_splitter.py
│   │       ├── fold_stratifier.py          #     5-fold cross-validation
│   │       └── test_splitter.py            #     Hold-out test split
│   │
│   ├── model/                              # Neural network models
│   │   │
│   │   ├── abstract.py
│   │   ├── full_model.py                   #   nn.Module combining architecture + behavior
│   │   ├── model_config.py
│   │   ├── architectures/
│   │   │   │
│   │   │   ├── MLP/                        #     Multi-layer perceptron (Linear)
│   │   │   ├── LeNet/                      #     1D CNN (Conv1d)
│   │   │   ├── PatchTST/                   #     Patch Time Series Transformer
│   │   │   ├── ResNet/                     #     Residual network (Conv1d blocks)
│   │   │   └── Transformer/                #     (empty — not yet implemented)
│   │   │
│   │   └── behavior/
│   │       │
│   │       └── classification_behavior.py  #   Cross-entropy loss, softmax, argmax
│   │
│   ├── training/                           # Training loop
│   │   │
│   │   ├── trainer.py                      #   Builds criterion, optimizer, runs fit()
│   │   ├── train_logic.py                  #   Epoch loop, val, checkpointing
│   │   ├── checkpoint.py                   #   Best-model early stopping on val loss
│   │   ├── train_display.py
│   │   └── training_config.py
│   │
│   ├── metrics/                            # Evaluation
│   │   │
│   │   ├── evaluation.py
│   │   ├── metrics_config.py
│   │   └── classification_metrics/         #   11 metric implementations
│   │       │
│   │       ├── abstract.py
│   │       ├── roc_auc.py
│   │       ├── pr_auc.py
│   │       ├── precision.py
│   │       ├── recall.py
│   │       ├── specificity.py
│   │       ├── f1_score.py
│   │       ├── accuracy.py
│   │       ├── macro_accuracy.py
│   │       ├── macro_f1_score.py
│   │       ├── macro_precision.py
│   │       └── macro_recall.py
│   │
│   ├── persistence/                        # Save/load experiment artifacts ("mpkg")
│   │   │
│   │   ├── persistence.py
│   │   ├── persistence_config.py
│   │   └── processes/                      #   Sub-persistence modules
│   │       │
│   │       ├── configuration_persistence.py
│   │       ├── weights_persistence.py
│   │       ├── json_persistence.py
│   │       ├── figures_persistence.py
│   │       └── model_output_persistence.py
│   │
│   └── gallery/                            # Explainability / visualization
│       │
│       ├── grad_cam.py                     #   For Conv1d models
│       ├── legrad.py                       #   For attention-based models
│       └── model_output.py                 #   Prediction + confidence + heatmap per example
│
├── config_plan/                            # Pre-built config instances
│
├── scripts/
│   ├── training/                           # Standalone training scripts & orchestrator
│   └── analysis/                           # EDA, visualization, metrics recomputation
│
├── tests/                                  # Pytest suite
│
├── modules/                                # Shared utilities
│
├── data/Elderly_Cough_Audio/               # Source dataset (gitignored)
│
└── outputs/                                # Experiment results (gitignored)
```

## Prerequisites

- Python 3.11
- CUDA-capable GPU (PyTorch is built against CUDA 12.4)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
uv sync
uv run python main.py
```

## Running Experiments

- `main.py` — default experiment
- `scripts/training/train_*.py` — individual data-pipeline × model combos
- `scripts/training/scripts_orchestrator.py` — batch run queued scripts
- `scripts/analysis/recompute_mpkg_metrics.py` — re-evaluate saved model packages

## Dataset

The Elderly Cough Audio dataset lives in `data/Elderly_Cough_Audio/` (gitignored).

## Development

See `AGENTS.md` and `CODING_PRINCIPLES.md`.
