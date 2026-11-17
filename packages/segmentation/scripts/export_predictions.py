"""Export whole-scene contrail probability maps for a pretrained bundle.

The MTG model is trained on 512 px windows of 1024 px frames, so the numbers worth
reporting must come from whole frames predicted the way users will: the library's
``predict`` cuts a scene larger than the model's native size into overlapping
native-size tiles and blends them with tent weights (the pixel scale never changes).
This script writes one float32 probability map per scene of a split; ``evaluate_scenes.py``
then scores those maps against the dataset labels.

The false-colour recipe comes from the training ``--config`` (its ``data`` section), not
from the bundle: the bundle may have been trained on another sensor's band keys.

Usage
-----
::

    uv run python packages/segmentation/scripts/export_predictions.py \
        --bundle /path/to/bundle \
        --config packages/segmentation/configs/mtg.yaml \
        --split test --out runs/mtg/preds_test

    # a quick smoke test on the first ten scenes, no TTA, on the CPU
    uv run python packages/segmentation/scripts/export_predictions.py \
        --bundle hf:org/name --split test --out /tmp/preds \
        --limit 10 --no-tta --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from deepcontrail.core.contract import read_band_manifest
from deepcontrail.core.false_color import false_color_from_spectral
from deepcontrail.segmentation import (
    SegmentationDataset,
    load_pretrained,
    predict,
)
from deepcontrail.segmentation.training import load_configs

if TYPE_CHECKING:
    from deepcontrail.segmentation import SegmentationDataConfig

logger = logging.getLogger(__name__)

# Position of each split in the tuple returned by ``load_configs``.
_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


def _split_dirname(split: str) -> str:
    """On-disk directory of a split's scenes: ``test/`` for the test split, ``train/`` else.

    Mirrors the dataset, where both ``train`` and ``val`` read from ``train/`` and only
    ``test`` reads from ``test/``.
    """
    return "test" if split == "test" else "train"


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write ``path`` atomically: a temp file in the same directory, then a rename.

    A crash mid-write leaves the previous map (or nothing) in place, never a half map, so
    a rerun can resume from where it stopped.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, array)
    tmp.replace(path)


def _select_data_config(
    config_path: str | Path, overrides: dict[str, Any] | None, split: str
) -> SegmentationDataConfig:
    """Return the ``SegmentationDataConfig`` of ``split`` from the training YAML.

    Index 0/1/2 of ``load_configs`` is train/val/test. ``overrides`` (e.g. ``{"data_root":
    ...}``) is forwarded to ``load_configs`` so ``--data-root`` overrides ``data.root``.
    """
    train_cfg, val_cfg, test_cfg, *_ = load_configs(config_path, overrides)
    return (train_cfg, val_cfg, test_cfg)[_SPLIT_INDEX[split]]


def main(argv: list[str] | None = None) -> int:
    """Export one float32 probability map per scene of a split; return 0."""
    parser = argparse.ArgumentParser(
        description="Export whole-scene probability maps for a pretrained bundle."
    )
    parser.add_argument("--bundle", required=True, help="Pretrained bundle: local folder or HF id.")
    parser.add_argument("--revision", default=None, help="Hub revision when --bundle is an id.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training YAML; its data section gives root, false_color, val_ids, pixel_shift.",
    )
    parser.add_argument(
        "--split", required=True, choices=["train", "val", "test"], help="Split to export."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the maps.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root overriding data.root of the config.",
    )
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply D4 TTA (default: the bundle's config.tta).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Composites per forward pass.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda when available, else cpu).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N scenes (smoke tests)."
    )
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = _select_data_config(args.config, {"data_root": args.data_root}, args.split)
    root = Path(data_cfg.root)
    bands_dir = root / "spectral_bands" / _split_dirname(args.split)
    manifest = read_band_manifest(root)

    pretrained = load_pretrained(args.bundle, revision=args.revision, device=device)
    stems = SegmentationDataset(data_cfg).stems
    if args.limit is not None:
        stems = stems[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    n = len(stems)
    tta = args.tta if args.tta is not None else pretrained.config.tta
    logger.info("Exporting %d scene(s) of the %s split of %s", n, args.split, root)

    done = 0
    for i, stem in enumerate(stems):
        out_path = args.out / f"{stem}.npy"
        if out_path.exists():  # resumable: keep the map already written
            done += 1
            continue
        bands = np.load(bands_dir / f"{stem}.npy")
        composite = false_color_from_spectral(bands, data_cfg.false_color, manifest)
        probs = predict(
            pretrained,
            composite,
            tta=tta,
            pixel_shift=data_cfg.pixel_shift,
            batch_size=args.batch_size,
        )
        _atomic_save_npy(out_path, np.ascontiguousarray(probs, dtype=np.float32))
        done += 1
        if i % 50 == 0:
            print(f"[{i + 1}/{n}] {stem} -> {out_path.name}")

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "bundle": str(args.bundle),
                "revision": args.revision,
                "config": str(args.config),
                "split": args.split,
                "tta": tta,
                "n_scenes": n,
                "created": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Exported {done}/{n} scene map(s) to {args.out} (manifest written)")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
