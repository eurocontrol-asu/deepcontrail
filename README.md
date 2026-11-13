# DeepContrail

[![PyPI](https://img.shields.io/pypi/v/deepcontrail-segmentation?label=PyPI)](https://pypi.org/project/deepcontrail-segmentation/)
[![Python](https://img.shields.io/pypi/pyversions/deepcontrail-segmentation)](https://pypi.org/project/deepcontrail-segmentation/)
[![CI](https://github.com/eurocontrol-asu/deepcontrail/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eurocontrol-asu/deepcontrail/actions/workflows/ci.yml)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24130966.svg)](https://doi.org/10.5281/zenodo.24130966)

Open-source library for **contrail detection in satellite imagery**. It does semantic
segmentation of contrails in GOES-16 imagery, with instance segmentation in preview; tracking,
flight attribution and altitude estimation are planned (see [Goal](#goal)).

This repository will host every phase of the project. **The current release ships the first phase:
semantic segmentation** of contrails in GOES-16 ABI imagery, trained and evaluated on the
OpenContrails dataset, plus a preview of instance segmentation. The later phases will be added as
they are developed.

**Status: alpha.** The API may change between minor versions. The packages support Python 3.11
to 3.13 and are tested on Linux. Training is meant for a GPU; prediction also runs on a CPU.

## Goal

Provide a clean, well-typed, and reusable library for detecting contrails in satellite imagery.
Development follows a phased roadmap:

1. **Semantic segmentation**: detect contrail pixels in GOES-16 imagery (**released in v0.1.0**)
2. **Instance segmentation**: separate individual contrails (**preview in v0.1.0**, tuned on
   GOES-16; MTG next)
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
| [`deepcontrail-instance`](packages/instance) | `deepcontrail.instance` | Instance segmentation (preview; the API may change): splits a segmentation map into individual contrails (Mannstein pipeline, learned scores); torch-free |
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

The released model scored **0.70414 private / 0.69328 public** on the hidden test set of the
Kaggle *Google Research – Identify Contrails to Reduce Global Warming* competition (TTA,
threshold 0.5). Kaggle's labels follow the older `20230419` export,
which differs from `20250305` (see [Data notes](#data-notes)).

## Installation

The packages are on PyPI; installing one installs `deepcontrail-core` too:

```bash
pip install deepcontrail-segmentation               # semantic segmentation, pretrained models
pip install "deepcontrail-segmentation[mlflow]"     # + the optional MLflow logger
pip install "deepcontrail-instance[map,scores]"     # instance segmentation (preview)
pip install deepcontrail-opencontrails              # the OpenContrails converter (TensorFlow)
```

For development, clone the repository and let [uv](https://docs.astral.sh/uv/) install the
workspace:

```bash
git clone https://github.com/eurocontrol-asu/deepcontrail.git
cd deepcontrail
uv sync                    # all packages, the dev tools and the training scripts' dependencies
uv sync --group convert    # + the OpenContrails converter (TensorFlow)
```

## Quickstart

### Predict with the pretrained model

Five sample scenes are published as the Hugging Face dataset
[`eurocontrol-asu/deepcontrail-samples`](https://huggingface.co/datasets/eurocontrol-asu/deepcontrail-samples)
(31 MB, OpenContrails test scenes in the dataset format), so this runs in a minute after
`pip install deepcontrail-segmentation`:

```python
import numpy as np
from huggingface_hub import snapshot_download
from deepcontrail.core import false_color_from_spectral, read_band_manifest
from deepcontrail.segmentation import load_pretrained, predict

root = snapshot_download("eurocontrol-asu/deepcontrail-samples", repo_type="dataset")
model = load_pretrained("eurocontrol-asu/deepcontrail-segmentation", revision="v0.1.0")
bands = np.load(f"{root}/spectral_bands/test/1801360214472457069.npy")
composite = false_color_from_spectral(bands, model.config.false_color, read_band_manifest(root))
probabilities = predict(model, composite, tta=model.config.tta)  # (256, 256), float32 in [0, 1]
mask = probabilities > model.config.threshold
```

With your own data, `root` is any dataset in the DeepContrail format (see [Data](#data)).
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
[OpenContrails page](https://sites.research.google/gr/contrails/public-datasets/), under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence. They are in the public Google
Cloud Storage bucket `goes_contrails_dataset`; download the `20250305` export with the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install):

```bash
gcloud storage cp -r gs://goes_contrails_dataset/20250305 .
```

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

## Contributing

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security problems privately, as described in
[SECURITY.md](SECURITY.md). Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## Use of AI tools

DeepContrail was developed with the help of [Claude Code](https://claude.com/claude-code), an AI
coding assistant by Anthropic, used for writing code, tests and documentation and for reviewing
them. The maintainers directed the design and the work, and reviewed, tested and merged
every change through pull requests; they are responsible for its content. The experiments and
the reported results come from training and evaluation runs, not from the assistant.

## Citing

If you use DeepContrail, please cite it
([doi:10.5281/zenodo.24130966](https://doi.org/10.5281/zenodo.24130966), all versions); GitHub's
"Cite this repository" button gives the reference from [CITATION.cff](CITATION.cff). If you use the pretrained model or the OpenContrails
data, please also cite the dataset:

```bibtex
@article{Ng2024ContrailDO,
  title={Contrail Detection on GOES-16 ABI With the OpenContrails Dataset},
  author={Joe Yue-Hei Ng and Kevin McCloskey and Jian Cui and Vincent R. Meijer and Erica Brand and Aaron Sarna and Nita Goyal and Christopher Van Arsdale and Scott Geraedts},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  volume={62},
  pages={1-14}
}
```

DeepContrail continues an earlier study of contrail segmentation on GOES-16 by the same unit:

```bibtex
@inproceedings{jarry2024deep,
  author    = {Jarry, Gabriel and Very, Philippe and Heffar, Amine and Torjman-Levavasseur, Valentin},
  title     = {Deep Semantic Contrails Segmentation of {GOES-16} Satellite Images: A Hyperparameter Exploration},
  booktitle = {Proceedings of the SESAR Innovation Days},
  year      = {2024},
  url       = {https://sesar.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_028%20final.pdf}
}
```

## Acknowledgements

The models are trained on the [OpenContrails](https://sites.research.google/gr/contrails/public-datasets/)
dataset by Google Research. GOES-16 imagery is courtesy of NOAA. The segmentation models build on
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch),
[timm](https://github.com/huggingface/pytorch-image-models) and
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). The instance package's
references are listed in [`packages/instance`](packages/instance#references-and-acknowledgements).
DeepContrail continues the contrail segmentation study of Jarry, Very, Heffar and
Torjman-Levavasseur ([SESAR Innovation Days 2024](https://sesar.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_028%20final.pdf); see [Citing](#citing)).

## Contact

DeepContrail is developed by the EUROCONTROL Aviation Sustainability Unit. Use
[GitHub issues](https://github.com/eurocontrol-asu/deepcontrail/issues) for bugs and questions, or
email [github-asu@eurocontrol.int](mailto:github-asu@eurocontrol.int).

## License

The code is licensed under the [European Union Public Licence v1.2 (EUPL-1.2)](LICENSE). The
pretrained weights are released under the Apache License 2.0; see their
[model card](https://huggingface.co/eurocontrol-asu/deepcontrail-segmentation).
