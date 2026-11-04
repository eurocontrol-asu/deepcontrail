"""Tests for scripts.optimize_mannstein (Mannstein search on train annotator samples)."""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from PIL import Image

from deepcontrail.instance.config import MannsteinConfig

if TYPE_CHECKING:
    from pathlib import Path

H = W = 64


requires_map = pytest.mark.skipif(
    importlib.util.find_spec("pycocotools") is None, reason="needs deepcontrail-instance[map]"
)


def _line(row: int, h: int = H, w: int = W) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[row, 8 : w - 8] = True
    return mask


def _make_annotator_root(tmp_path: Path, n_scenes: int) -> Path:
    """Train split where scene i has 2 annotator slots; slot 0 drew one line, slot 1 nothing.

    Every third scene has no contrail at all.
    """
    for sub in ("annotator_masks", "instance_masks", "semantic_masks"):
        (tmp_path / sub / "train").mkdir(parents=True)
    for i in range(n_scenes):
        stem = f"scene_{i:03d}"
        inst = np.zeros((1, H, W, 2), dtype=np.int8)
        ann = np.zeros((H, W, 2), dtype=np.uint8)
        if i % 3:
            line = _line(10 + 3 * (i % 12))
            inst[0, :, :, 0] = line
            inst[0, :, :, 1] = -1
            ann[..., 0] = line
        np.save(tmp_path / "instance_masks/train" / f"{stem}.npy", inst)
        np.save(tmp_path / "annotator_masks/train" / f"{stem}.npy", ann)
        Image.fromarray(ann[..., 0] * 255, mode="L").save(
            tmp_path / "semantic_masks/train" / f"{stem}.png"
        )
    return tmp_path


# --- stem selection (shared with optimize_pipeline) ---


def test_write_best_yaml_roundtrips_through_from_yaml(tmp_path: Path) -> None:
    from instance_scripts.optimize_mannstein import write_best_yaml

    config = MannsteinConfig(num_kernels=16, kernel_size=9, filter_avg_size=3)
    path = tmp_path / "best_mannstein.yaml"

    write_best_yaml(config, path)

    assert "mannstein" in yaml.safe_load(path.read_text())
    assert MannsteinConfig.from_yaml(path) == config


# --- scoring ---


@requires_map
def test_main_writes_best_config_and_holdout_summary(tmp_path: Path) -> None:
    from instance_scripts.optimize_mannstein import main, parse_args

    root = _make_annotator_root(tmp_path / "data", 18)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.dump({"mannstein": MannsteinConfig(min_pixels=7).model_dump()}))
    out = tmp_path / "out"

    summary = main(
        parse_args(
            ["--root", str(root), "--n-samples", "4", "--n-holdout", "4", "--n-trials", "2",
             "--workers", "1", "--baseline-configs", str(base), "--output-dir", str(out)]
        )
    )  # fmt: skip

    assert MannsteinConfig.from_yaml(out / "best_mannstein.yaml") == MannsteinConfig.from_dict(
        summary["best_config"]
    )
    written = json.loads((out / "summary.json").read_text())
    assert set(written["holdout"]) == {"baseline_0", "best"}
    assert "map" in written["holdout"]["best"]


# --- probability-map input ---


def _keep_only_instance_masks(root: Path) -> None:
    import shutil

    for sub in ("annotator_masks", "semantic_masks", "probability_masks"):
        shutil.rmtree(root / sub, ignore_errors=True)


@requires_map
def test_main_needs_only_instance_masks(tmp_path: Path) -> None:
    from instance_scripts.optimize_mannstein import main, parse_args

    root = _make_annotator_root(tmp_path / "data", 18)
    _keep_only_instance_masks(root)
    out = tmp_path / "out"

    main(
        parse_args(
            ["--root", str(root), "--n-samples", "4", "--n-holdout", "4", "--n-trials", "2",
             "--workers", "1", "--output-dir", str(out)]
        )
    )  # fmt: skip

    assert (out / "summary.json").exists()
