# Changelog

All notable changes to DeepContrail are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). All packages share one version number.

## [Unreleased]

### Changed

- `deepcontrail-instance` is no longer a preview: it has the same status as the other packages
  (alpha: the API may change between minor versions, and every change is listed here).
- `instance_pipeline.yaml` and `score_model.json` written by a newer `deepcontrail-instance` (a
  higher `format_version`) are refused with a message to upgrade. Any new field, or change of
  meaning, in these files raises their `format_version`.
- The root README shows the instance results and, in the quickstart, how to separate the
  contrails. The instance README explains what a score means, corrects the crossing-angle limit
  (contrails crossing at less than about 25° often merge) and says which instance step is
  published.

### Removed

- `deepcontrail.instance.instance_features`: it found the contrail among the scene's masks by
  object identity, so an equal copy of a mask got a wrong `shared_pixels_frac`. Use
  `scene_features(masks, probabilities, scores)`, now public, which describes every detection of
  a scene.

## [0.2.0] - 2026-11-13

Fixes and clearer behaviour on top of 0.1.0, found after the first release. The pretrained model
is unchanged (Hugging Face revision `v0.1.0`).

### Added

- `deepcontrail.core.scene_stems(directory, suffix)`: the ids of one dataset directory, skipping
  hidden files.
- Five sample scenes, published as the Hugging Face dataset `eurocontrol-asu/deepcontrail-samples`:
  the README quickstart now runs in a minute, without downloading the full dataset.

### Changed

- `predict` follows the model's `config.tta` when `tta` is not given (it defaulted to no
  test-time augmentation, while the released model recommends it). Pass `tta=False` for the
  previous behaviour.
- Dataset format: unknown fields in `bands.json` and `dataset_info.json` are now ignored with a
  warning naming them, instead of rejected, so optional fields can be added within version 1;
  a dataset of a newer format version gets an error asking to upgrade `deepcontrail-core`. The
  versioning rule is documented in the `deepcontrail-core` README.
- `deepcontrail.instance.refine.grow_instances` grows with 8-connectivity by default, like the
  instance pipeline (it defaulted to 4). The pipeline and the published instance step are
  unchanged.
- Instance scoring is much faster on busy scenes (11 s → 0.9 s for 613 detections), with
  identical features and scores.
- Package metadata names the author and the maintaining unit; the packages are published by the
  `eurocontrol-asu` PyPI organisation.
- `segmentation-models-pytorch` 0.6 is supported (the released model predicts bit-identically).

### Fixed

- `save_instance_step` without a score model removes an earlier `score_model.json` from the
  folder; it was left behind and still loaded.
- Hidden files, such as the `._<name>` companions macOS writes when copying a dataset, were
  counted as scenes by the validator, the segmentation dataset, the instance evaluation and the
  converter.
- The deprecated albumentations `ShiftScaleRotate` (removed in albumentations 3) is replaced by the
  equivalent `Affine`; augmentations are identical for the same seed.

### Removed

- `resolve_run_paths`, `CHECKPOINT_ROOT_ENV` and `MLFLOW_TRACKING_URI_ENV` are no longer exported
  by `deepcontrail.segmentation.training`; they served only the repository's training script,
  where they now live.

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

[Unreleased]: https://github.com/eurocontrol-asu/deepcontrail/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/eurocontrol-asu/deepcontrail/releases/tag/v0.2.0
[0.1.0]: https://github.com/eurocontrol-asu/deepcontrail/releases/tag/v0.1.0
