# Changelog

All notable changes to DeepContrail are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). All packages share one version number.

## [Unreleased]

### Changed

- `predict` follows the model's `config.tta` when `tta` is not given (it defaulted to no
  test-time augmentation, while the released model recommends it). Pass `tta=False` for the
  previous behaviour.

## [0.1.0] - 2026-11-11

The first release: semantic segmentation of contrails in GOES-16 ABI imagery, and a preview of
instance segmentation.

### Added

- `deepcontrail-core` (torch-free): the dataset format v1 (`bands.json`, `dataset_info.json`)
  and its validator `deepcontrail-validate`; false-colour composites built from band keys;
  train/validation/fold splits; Dice and IoU.
- `deepcontrail-segmentation`: dataset, models (segmentation-models-pytorch with timm encoders,
  including Unet++ with transformer-style encoders) and losses; a Lightning training entry point,
  `train()` / `deepcontrail-train`, with csv or MLflow logging; D4 test-time augmentation;
  pretrained models (`load_pretrained`, `predict`).
- `deepcontrail-instance` (preview; the API may change): splits a segmentation map into
  individual contrails and evaluates them with COCO metrics.
- `deepcontrail-opencontrails` (a tool): converts the OpenContrails TFRecords to the dataset
  format.
- Pretrained model [`eurocontrol-asu/deepcontrail-segmentation`](https://huggingface.co/eurocontrol-asu/deepcontrail-segmentation),
  revision `v0.1.0`: Unet++ with a CAFormer-B36 encoder, test Dice 0.6974 with test-time
  augmentation, plus the instance step tuned for it.

[Unreleased]: https://github.com/eurocontrol-asu/deepcontrail/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/eurocontrol-asu/deepcontrail/releases/tag/v0.1.0
