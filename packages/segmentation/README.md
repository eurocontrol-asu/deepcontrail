# deepcontrail-segmentation

Semantic segmentation of contrails, part of
[DeepContrail](https://github.com/eurocontrol-asu/deepcontrail): the segmentation dataset,
transforms, model factory (segmentation-models-pytorch with timm encoders), losses, a Lightning
module, D4 test-time augmentation, a training entry point, and pretrained models.

It reads datasets in the DeepContrail format (see [`deepcontrail-core`](../core)).
`import deepcontrail.segmentation` stays light; torch is loaded by the parts that need it.

Unet++ also works with transformer-style encoders such as CAFormer, which have no feature map at
half the input resolution: `build_model` adds a small convolutional stem for that level (two 3×3
convolutions, 64 channels), which segmentation-models-pytorch's Unet++ needs and cannot build itself.

## Pretrained models

A pretrained model is a folder (or Hugging Face Hub repository) with `model.safetensors` and a
`config.json` holding everything needed to run it: architecture, composite recipe by band key,
normalisation, input size, decision threshold and whether test-time augmentation is recommended.

```python
from deepcontrail.core import false_color_from_spectral, read_band_manifest
from deepcontrail.segmentation import load_pretrained, predict

model = load_pretrained("path/to/bundle")  # or "org/repo" on the Hub, with revision="..."
composite = false_color_from_spectral(bands, model.config.false_color, read_band_manifest(root))
probabilities = predict(model, composite, tta=model.config.tta)
mask = probabilities > model.config.threshold
```

`predict` takes one `(H, W, 3)` composite or a batch `(N, H, W, 3)` and returns probabilities at
the input's resolution. It applies exactly the training preprocessing (resize, normalisation);
`pixel_shift=` aligns inputs that follow another label convention (0.5 for Kaggle data with a
`20250305` model).

A bundle may also carry the instance step tuned for its model (`instance_pipeline.yaml`,
`score_model.json`): `deepcontrail.instance.load_instance_step(model.directory)` reads it, and
`scripts/export_pretrained.py --instance-pipeline ... --score-model ...` adds it.

## Training

Lightning's `Trainer` runs the training; the entry point builds it from an experiment YAML
(`data`, `model`, `loss`, `training` sections; see [`configs/experiment.yaml`](configs/experiment.yaml)):

```bash
deepcontrail-train experiment.yaml --data-root /path/to/dataset --output-dir runs \
    [--run-name NAME] [--logger csv|mlflow] [--epochs N] [--accelerator gpu]
```

```python
from deepcontrail.segmentation import train

run = train("experiment.yaml", output_dir="runs")
print(run.best_checkpoint, run.best_val_dice)
```

Each run writes into `runs/<run-name>/`: the best checkpoint by validation Dice and the last
one, `split/train.txt` and `split/val.txt` (the exact records used), and the logger's files. The
logger is `training.logger` in the config: `csv` (default, no extra package) or `mlflow` (install
`deepcontrail-segmentation[mlflow]`; tracking URI from `MLFLOW_TRACKING_URI`, else
`<output-dir>/mlflow.db`). Metrics: train and validation loss, Dice and IoU per epoch, the
per-step loss and the learning rate.

`prepare_training(...)` builds the same run without training it.

## Repository scripts (not installed with the package)

- **`scripts/train.py`**: the project's experiment workflow on top of the entry point: MLflow
  tracking (parameters, curves figure, dataset metadata, split lists, best checkpoint), then a
  threshold search on validation and evaluation on the test split, with and without D4 TTA.
  Where runs go is set once in a `.env` at the repository root (`cp .env.example .env`):
  `DEEPCONTRAIL_CHECKPOINT_ROOT` and `MLFLOW_TRACKING_URI`.

  ```bash
  uv run python packages/segmentation/scripts/train.py --data-root /path/to/dataset --run-name my-run
  ```
- **`scripts/evaluate.py`**: evaluate a checkpoint (`--checkpoint`) or an ensemble
  (`--checkpoints a.ckpt b.ckpt ...`) on validation and test, with `--tta` and `--threshold`.
- **`scripts/export_pretrained.py`**: turn a checkpoint into a pretrained bundle, check that it
  reproduces the checkpoint exactly, and write a draft model card:

  ```bash
  uv run python packages/segmentation/scripts/export_pretrained.py \
      --checkpoint run/best.ckpt --output-dir bundle --metric test_dice_tta_at_0.5=0.6960
  ```

The reference configuration is [`configs/experiment.yaml`](configs/experiment.yaml).
