# core/data_pipeline — Audio → Feature → Dataset

Turns the raw Elderly Cough Audio dataset into train/validation/test `ExampleDataset`
objects. Everything is config-driven through five pipeline stages.

## Position in the framework

```
metadata.xlsx + translations.json + source_data/ audio files
        │
        ▼
ElderlyCoughAudioSourceReader  ──► SourceSeries (one raw recording)
        │ 
        │ segment
        │ 
CoughSegmenter                 ──► Example (one cough segment)
        │ 
        │ transform
        │ 
Raw | MFCC | MelBand           ──► Example (feature frames)
        │ 
        │ pad
        │ 
ZeroPadder                     ──► Example (fixed length)
        │ 
        │ split
        │ 
DataSplitter                   ──► DataSplit (test + 5 development folds)
        │
        ▼
ExperimentOrchestrator.train_model()  (one fold at a time)
```

## Key classes

| Class | Role |
|---|---|
| `DataPipeline` | Compose and run the five stages; entry points `get_examples()`, `get_dataset()`, `get_data_split()` |
| `DataPipelineConfig` | Bundle of the five stage instances; `default()` builds the standard stack |
| `SourceReader`, `Segmenter`, `Transformer`, `Padder`, `Splitter` | ABCs defining the stage contracts |
| `ElderlyCoughAudioSourceReader` | Dataset-specific I/O: metadata + audio → `SourceSeries` |
| `CoughSegmenter`, `MFCC`, `MelBand`, `ZeroPadder` | Concrete preprocessing stages |
| `DataSplitter` | Splits examples into hold-out test + 5-fold development sets |
| `ExampleDataset` | `torch.utils.data.Dataset` wrapper; yields `{"value", "label", "metadata"}` |
| `SourceSeries`, `Example`, `DevelopmentFold`, `DataSplit` | Intermediary dataclasses flowing through the pipeline |

## Core concepts

- **Source read** — reads `metadata.xlsx` (worksheet `"dynamo"`), type-casts each
  row (`MetadataTypeCaster`), applies `translations.json` string replacements
  (`MetadataTranslator`), decodes and resamples audio to **16 kHz**
  (`RawAudioReader` + `AudioResampler`, with a SHA-256-keyed on-disk cache).
- **Segmentation** — each recording is sliced into cough segments using the
  `detected_cough_segments` metadata intervals (`value[start:end+1]`).
- **Transform** — `MFCC` produces log-MFCC frames `(time, n_mfcc)`; `MelBand`
  produces log-mel band energies `(time, n_mels)`. Same STFT settings:
  `n_fft=400`, `hop_length=160` (10 ms), `n_mels=40`.
- **Padding** — `ZeroPadder` pads the time axis to **820 frames** (~8.2 s of
  context) in `left`/`right`/`balanced`/`random` modes.
- **Splitting** — patient-grouped and stratified: a hold-out **test split**
  (`test_ratio=0.1`) and **5-fold cross-validation** via
  `StratifiedGroupKFold`, grouped by `patient_id` so every segment of a patient
  stays in one fold.

## Usage

The two shipped pipeline variants live in `config_plan/data_pipeline/`
(`mfcc_data_pipeline_config`, `melband_data_pipeline_config`) and are imported by
scripts. For a quick standalone pipeline:

```python
from core.data_pipeline import DataPipeline, DataPipelineConfig

config = DataPipelineConfig.default()
pipeline = DataPipeline.create(config)
data_split = pipeline.get_data_split()   # test_dataset + development_folds
```

## Key parameters (`DataPipelineConfig.default()`)

- Source: `data/Elderly_Cough_Audio/` — `metadata.xlsx`, `translations.json`, `source_data/`
- MFCC/MelBand: `sample_rate=16000`, `n_fft=400`, `win_length=400`, `hop_length=160`, `n_mels=40`
- ZeroPadder: `target_length=820`, `padding_type="random"`, `random_seed=42`
- DataSplitter: `group_metadata_key="patient_id"`, `test_ratio=0.1`, `number_of_folds=5`, `random_seed=42`

## Gotchas

- **Patient exclusions** — `patient_id_exception.py` lists patients dropped at read
  time: mixed medical conditions, empty cough segments, mixed labels (incompatible
  with patient-grouped splitting). Rows with `audio_exists=False` or
  `usability=False` are also dropped.
- **Label mapping** — `is_infectious=True → 1`, else `0`; the original string label
  is preserved under the metadata key `"original_label"`.
- **Sampling stage is a stub** — `KeepAllSample` is an empty placeholder; there is
  no `Sampler` stage yet.
- **`test_dataset` is currently unused** — `ExperimentOrchestrator.test_model()`
  is a stub, so training only consumes the development folds.
- Segmentation slices are inclusive of the end index (`value[start:end+1]`).

## Tests

`tests/data_pipeline/` covers `ExampleDataset` and the feature transformers. The
source reader, segmentation, padding, and stratifier are untested.