# DeepContrail

Open-source library for **contrail detection, segmentation, tracking, and flight attribution**
in satellite imagery.

This repository will host every phase of the project. **Version 0.1.0 ships the first phase:
semantic segmentation** of contrails in GOES-16 ABI imagery, trained and evaluated on the
OpenContrails dataset. The later phases will be added as they are developed.

## Goal

Provide a clean, well-typed, and reusable library for detecting contrails in satellite imagery.
Development follows a phased roadmap:

1. **Semantic segmentation**: detect contrail pixels in GOES-16 imagery (**released in v0.1.0**)
2. **Instance segmentation**: separate individual contrails (MTG) (next)
3. **Tracking**: follow contrails across time steps
4. **Attribution**: assign ADS-B flight trajectories to detected contrails
5. **Altitude estimation**: infer contrail altitude from imagery

## Packages

The repository is a [uv](https://github.com/astral-sh/uv) workspace of separately installable
packages that share the `deepcontrail` namespace:

| Package | Import | Contents |
|---|---|---|
| [`deepcontrail-core`](packages/core) | `deepcontrail.core` | Torch-free foundations: the dataset format and its validation, false-colour composites, train/val/fold splits, Dice/IoU |
| [`deepcontrail-segmentation`](packages/segmentation) | `deepcontrail.segmentation` | Semantic segmentation: dataset, models, losses, training (`train`), pretrained models (`load_pretrained`, `predict`) |
| [`deepcontrail-instance`](packages/instance) | `deepcontrail.instance` | Instance segmentation (in progress): splits a segmentation map into individual contrails (Mannstein pipeline, learned scores); torch-free |
| [`deepcontrail-opencontrails`](tools/opencontrails) | `deepcontrail.opencontrails` | Converter from the OpenContrails TFRecords to the dataset format (a tool, not part of the library) |

## Results: semantic segmentation

Reference model: Unet++ with a CAFormer-B36 encoder at 512 px, trained 40 epochs on the
OpenContrails `20250305` export with
[`packages/segmentation/configs/experiment.yaml`](packages/segmentation/configs/experiment.yaml).
CAFormer has no feature map at half the input resolution, which Unet++ needs: `build_model` adds a
small convolutional stem for that level (see [`packages/segmentation`](packages/segmentation)).
Its weights are on the Hugging Face Hub as
[`eurocontrol-asu/deepcontrail-segmentation`](https://huggingface.co/eurocontrol-asu/deepcontrail-segmentation).
Global Dice on the held-out test split (1,879 scenes):

| Inference | Threshold | Test Dice |
|---|---|---|
| Single pass | 0.5 | 0.6890 |
| Single pass | tuned on validation | 0.6870 |
| D4 test-time augmentation (8×) | 0.5 | **0.6974** |
| D4 test-time augmentation (8×) | tuned on validation | 0.6937 |

GPU training is not bit-for-bit deterministic, so compare architectures over several runs (TTA,
threshold 0.5, same recipe). Before the augmentation fix of #47, when the two flips always fired
together, two Unet++ trainings scored 0.6992 and 0.6995 and four UPerNet trainings 0.6916–0.6960.
With the fix, Unet++ scored 0.6974 (the model above) and UPerNet 0.6912.

An earlier Unet++ training (before the augmentation fix) scored **0.6979 private / 0.6816 public**
on the hidden test set of the Kaggle *Google Research – Identify Contrails to Reduce Global
Warming* competition (TTA, threshold 0.5). Kaggle's labels follow the older `20230419` export,
which differs from `20250305` (see [Data notes](#data-notes)).

## Installation

From a clone (the packages are not on PyPI yet):

```bash
git clone https://github.com/eurocontrol-asu/deepcontrail.git
cd deepcontrail
uv sync                    # all packages, the dev tools and the training scripts' dependencies
uv sync --group convert    # + the OpenContrails converter (TensorFlow)
```

Or install the packages into an existing environment from git:

```bash
pip install \
    "deepcontrail-core @ git+https://github.com/eurocontrol-asu/deepcontrail#subdirectory=packages/core" \
    "deepcontrail-segmentation @ git+https://github.com/eurocontrol-asu/deepcontrail#subdirectory=packages/segmentation"
```

`deepcontrail-segmentation[mlflow]` adds the optional MLflow logger.

For instance segmentation, install core with it in the same command (pip does not read the
workspace sources, so `deepcontrail-core` must be given explicitly):

```bash
pip install \
    "deepcontrail-core @ git+https://github.com/eurocontrol-asu/deepcontrail#subdirectory=packages/core" \
    "deepcontrail-instance[map,scores] @ git+https://github.com/eurocontrol-asu/deepcontrail#subdirectory=packages/instance"
```

## Quickstart

### Predict with the pretrained model

```python
import numpy as np
from deepcontrail.core import false_color_from_spectral, read_band_manifest
from deepcontrail.segmentation import load_pretrained, predict

model = load_pretrained("eurocontrol-asu/deepcontrail-segmentation", revision="v0.1.0")
bands = np.load(f"{root}/spectral_bands/test/{record_id}.npy")
composite = false_color_from_spectral(bands, model.config.false_color, read_band_manifest(root))
probabilities = predict(model, composite, tta=True)  # (H, W), float32 in [0, 1]
mask = probabilities > model.config.threshold
```

`load_pretrained` also takes a local folder. The
[model card](https://huggingface.co/eurocontrol-asu/deepcontrail-segmentation) describes the
training, the scores and the limits.

### Train

```bash
uv run deepcontrail-train packages/segmentation/configs/experiment.yaml \
    --data-root /path/to/dataset --output-dir runs        # csv logger by default
```

or from Python: `deepcontrail.segmentation.train("experiment.yaml", output_dir="runs")`.
Lightning's `Trainer` does the training; each run writes its checkpoints, metrics and the exact
train/validation records to `runs/<run-name>/`.

The repository's experiment script adds MLflow tracking, test evaluation and test-time
augmentation; see [`packages/segmentation`](packages/segmentation).

## Data

The library reads one on-disk **dataset format** and nothing else: spectral arrays, label
masks and two small metadata files describing the bands and the dataset. Any data converted to
it works, from any sensor; the format and its validation are described in
[`packages/core`](packages/core). For the OpenContrails dataset the converter does it:

```bash
uv run deepcontrail-convert-opencontrails \
    --tfrecord-dir /path/to/20250305/tfrecords --output-dir /path/to/dataset
uv run deepcontrail-validate /path/to/dataset --deep
```

The raw TFRecords are published by Google Research on the
[OpenContrails page](https://sites.research.google/gr/contrails/public-datasets/), in the bucket
[`goes_contrails_dataset`](https://storage.cloud.google.com/goes_contrails_dataset) (export `20250305/`).

### Data notes

- **Ash RGB recipe.** The models use the standard ash RGB: R = B15 − B14 on [−4, 2] K,
  G = B14 − B11 on [−4, 5] K, B = B14 on [243, 303] K (GOES-16 ABI bands). In the OpenContrails
  TFRecords `data_10um` is ABI band **13**; band 11 is the `cloud_top` key. A green channel built
  from `data_10um` is not the standard recipe and scored 0.009–0.018 Dice lower.
- **Pixel shift.** In the `20230419` export the masks follow the Kaggle convention, half a pixel
  down-right of the images (`pixel_shift: 0.5`); in `20250305` they are aligned
  (`pixel_shift: 0.0`). Each converted dataset records its value in
  `metadata/dataset_info.json`. On Kaggle data, keep the 0.5 shift even with a `20250305` model.
- **The exports differ beyond the shift.** After alignment the `20250305` masks overlap the
  `20230419`/Kaggle ones at an IoU of about 0.78, so scores on the two are not directly comparable.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

All four must be green before opening a pull request. CI also tests each package in an
environment with only that package installed.

## License

The code is licensed under the [European Union Public Licence v1.2 (EUPL-1.2)](LICENSE). The
pretrained weights are released under the Apache License 2.0; see their
[model card](https://huggingface.co/eurocontrol-asu/deepcontrail-segmentation).
