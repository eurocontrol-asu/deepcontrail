"""Export a trained Lightning checkpoint as a pretrained bundle, with a draft model card.

Writes ``model.safetensors`` + ``config.json`` (loadable with
``deepcontrail.segmentation.load_pretrained``) and a ``README.md`` model card, then
checks that the bundle reproduces the checkpoint's outputs exactly.

Usage
-----
::

    uv run python packages/segmentation/scripts/export_pretrained.py \
        --checkpoint /path/to/run/best-val_dice=0.6963.ckpt \
        --config packages/segmentation/configs/experiment.yaml \
        --output-dir /path/to/bundle \
        --metric test_dice_tta_at_0.5=0.6960 --metric test_dice_at_0.5=0.6892

The licence and attribution sections of the card are left for a person to fill in
before publishing.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from deepcontrail.core.contract import DatasetContractError, read_band_manifest, read_dataset_info
from deepcontrail.segmentation.models import build_model
from deepcontrail.segmentation.pretrained import (
    PretrainedConfig,
    load_pretrained,
    save_pretrained,
)
from deepcontrail.segmentation.training import load_configs

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


def _parse_metric(text: str) -> tuple[str, float]:
    name, sep, value = text.partition("=")
    if not sep:
        msg = f"--metric expects name=value, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return name, float(value)


def _band_description(keys: tuple[str, ...], root: str) -> str:
    """Describe ``a - b`` by key, with the sensor's band IDs when the dataset has them."""
    text = " - ".join(keys)
    try:
        manifest = read_band_manifest(root)
    except DatasetContractError:
        return text
    bands = [manifest.channels[manifest.index_of(k)].sensor_band for k in keys]
    if all(bands):
        noun = "band" if len(bands) == 1 else "bands"
        text += f" ({manifest.sensor} {noun} {' - '.join(b for b in bands if b)})"
    return text


def _model_card(config: PretrainedConfig, data_root: str, name: str) -> str:
    """Draft README.md for the Hugging Face Hub."""
    channels = "\n".join(
        f"| {colour} | {_band_description(ch.bands, data_root)} | {list(ch.bounds)} |"
        for colour, ch in zip("RGB", config.false_color.channels, strict=True)
    )
    metrics = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in config.metrics.items()) or "| - | - |"
    provenance = "\n".join(f"| {k} | `{v}` |" for k, v in config.provenance.items())
    tta = "recommended (8 flips/rotations averaged)" if config.tta else "not used"
    return f"""---
library_name: deepcontrail
tags: [contrails, semantic-segmentation, remote-sensing, satellite]
license: TO FILL IN
---

# {name}

Contrail segmentation model for
[DeepContrail](https://github.com/eurocontrol-asu/deepcontrail): per-pixel contrail
probability in a satellite false-colour composite.

## Model

- **Architecture:** {config.model.architecture}, encoder `{config.model.encoder_name}`
- **Input:** a 3-channel composite in [0, 1] (below), resized to {config.image_size} px and
  normalised with mean {list(config.normalize.mean)}, std {list(config.normalize.std)}
- **Output:** contrail probability at the input's resolution; contrail where > {config.threshold}
- **Test-time augmentation:** {tta}

## Input composite

| Channel | Bands | Bounds (scaled to [0, 1], clipped) |
|---|---|---|
{channels}

## Usage

```python
from deepcontrail.core import false_color_from_spectral, read_band_manifest
from deepcontrail.segmentation import load_pretrained, predict

model = load_pretrained("<this repository>")
composite = false_color_from_spectral(bands, model.config.false_color, read_band_manifest(root))
probabilities = predict(model, composite, tta=model.config.tta)
mask = probabilities > model.config.threshold
```

`bands` is one `spectral_bands/<id>.npy` array of a dataset in the DeepContrail format.

## Evaluation

| Metric | Value |
|---|---|
{metrics}

## Provenance

| | |
|---|---|
{provenance}

## Licence and attribution

TO FILL IN before publishing: the licence of these weights, and the attribution
required by the training data's licence.
"""


def main(argv: list[str] | None = None) -> int:
    """Export the checkpoint, check the bundle against it, write the card; return 0."""
    parser = argparse.ArgumentParser(description="Export a checkpoint as a pretrained bundle.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Lightning .ckpt file.")
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="Experiment YAML of the run."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Bundle directory.")
    parser.add_argument(
        "--metric",
        type=_parse_metric,
        action="append",
        default=[],
        help="Evaluation result to record, name=value (repeatable).",
    )
    parser.add_argument("--run-name", default=None, help="Training run (default: ckpt folder).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold.")
    args = parser.parse_args(argv)

    data_cfg, _, _, model_cfg, *_ = load_configs(args.config)
    checkpoint: dict[str, Any] = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = {
        k.removeprefix("model."): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
    model = build_model(model_cfg.model_copy(update={"encoder_weights": None}))
    model.load_state_dict(weights, strict=True)
    model.eval()

    provenance = {
        "checkpoint": args.checkpoint.name,
        "run_name": args.run_name or args.checkpoint.parent.name,
        "deepcontrail_version": version("deepcontrail-segmentation"),
    }
    try:
        info = read_dataset_info(data_cfg.root)
        provenance |= {"data_source": info.source, "data_export": str(info.export)}
    except DatasetContractError:
        pass

    config = PretrainedConfig(
        format_version=1,
        model=model_cfg,
        false_color=data_cfg.false_color,
        normalize=data_cfg.normalize,
        image_size=data_cfg.image_size,
        native_size=int(checkpoint["hyper_parameters"]["native_size"]),
        threshold=args.threshold,
        metrics=dict(args.metric),
        provenance=provenance,
    )
    save_pretrained(model, config, args.output_dir)

    # The bundle must give exactly the checkpoint's outputs.
    loaded = load_pretrained(args.output_dir)
    x = torch.randn(1, 3, config.image_size, config.image_size, generator=torch.manual_seed(0))
    with torch.no_grad():
        if not torch.equal(loaded.model(x), model(x)):
            msg = "the exported bundle does not reproduce the checkpoint's outputs"
            raise RuntimeError(msg)

    name = args.run_name or args.checkpoint.parent.name
    (args.output_dir / "README.md").write_text(_model_card(config, data_cfg.root, name))
    print(f"Exported {args.checkpoint} -> {args.output_dir} (outputs identical to the checkpoint)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
