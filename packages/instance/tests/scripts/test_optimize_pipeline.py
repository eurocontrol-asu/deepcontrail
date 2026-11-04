"""Tests for scripts.optimize_pipeline (search over InstancePipelineConfig)."""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from deepcontrail.instance import InstancePipelineConfig

if TYPE_CHECKING:
    from pathlib import Path

H = W = 64


requires_map = pytest.mark.skipif(
    importlib.util.find_spec("pycocotools") is None, reason="needs deepcontrail-instance[map]"
)


def _line_probs(row: int) -> np.ndarray:
    """Probability map with one long horizontal contrail at ``row``."""
    probs = np.zeros((H, W), dtype=np.float32)
    probs[row, 8 : W - 8] = 0.9
    return probs


def _line_gt(row: int) -> np.ndarray:
    gt = np.zeros((H, W), dtype=bool)
    gt[row, 8 : W - 8] = True
    return gt


def _make_pred_root(tmp_path: Path, n: int) -> tuple[Path, Path]:
    """Dataset root with ``n`` train instance masks plus a folder of predictions."""
    inst_dir = tmp_path / "root" / "instance_masks" / "train"
    sem_dir = tmp_path / "root" / "semantic_masks" / "train"
    pred_dir = tmp_path / "preds"
    for d in (inst_dir, sem_dir, pred_dir):
        d.mkdir(parents=True)
    for i in range(n):
        stem = f"scene_{i:03d}"
        row = 10 + 4 * (i % 10)
        np.save(pred_dir / f"{stem}.npy", _line_probs(row).astype(np.float16))
        raw = np.zeros((1, H, W, 2), dtype=np.int8)
        raw[0, :, :, 0] = _line_gt(row).astype(np.int8)
        raw[0, :, :, 1] = -1
        np.save(inst_dir / f"{stem}.npy", raw)
        (sem_dir / f"{stem}.png").touch()
    return tmp_path / "root", pred_dir


@requires_map
def test_main_writes_loadable_best_config_and_summary(tmp_path: Path) -> None:
    from instance_scripts.optimize_pipeline import main, parse_args

    root, pred_dir = _make_pred_root(tmp_path, 8)
    out = tmp_path / "out"
    args = parse_args(
        [
            "--root", str(root), "--pred-dir", str(pred_dir), "--n-images", "4",
            "--n-holdout", "4", "--n-trials", "3", "--workers", "1", "--output-dir", str(out),
        ]
    )  # fmt: skip

    summary = main(args)

    best = InstancePipelineConfig.from_yaml(out / "best_pipeline.yaml")
    assert best == InstancePipelineConfig.from_dict(summary["best_config"])
    written = json.loads((out / "summary.json").read_text())
    assert set(written["holdout"]) == {"baseline", "baseline_grown", "defaults", "best"}, (
        "every starting config is re-scored on the holdout, including the shipped defaults"
    )
    assert isinstance(written["search_map"]["best"], float)
    assert "ap_per_threshold" in written["holdout"]["best"]
    assert written["search_size"] == 4
    assert written["holdout_size"] == 4


def _make_majority_pred_root(tmp_path: Path, n: int) -> tuple[Path, Path]:
    """Like ``_make_pred_root`` but 3 annotators, 2 of them drawing the line (a consensus)."""
    from PIL import Image

    inst_dir = tmp_path / "root" / "instance_masks" / "train"
    sem_dir = tmp_path / "root" / "semantic_masks" / "train"
    pred_dir = tmp_path / "preds"
    for d in (inst_dir, sem_dir, pred_dir):
        d.mkdir(parents=True)
    for i in range(n):
        stem = f"scene_{i:03d}"
        row = 10 + 4 * (i % 10)
        np.save(pred_dir / f"{stem}.npy", _line_probs(row).astype(np.float16))
        raw = np.full((1, H, W, 3), -1, dtype=np.int8)
        raw[0, :, :, 0] = _line_gt(row)
        raw[0, :, :, 1] = _line_gt(row)
        np.save(inst_dir / f"{stem}.npy", raw)
        Image.fromarray(_line_gt(row).astype(np.uint8) * 255, mode="L").save(
            sem_dir / f"{stem}.png"
        )
    return tmp_path / "root", pred_dir


@requires_map
def test_main_consensus_objective_scores_one_sample_per_scene(tmp_path: Path) -> None:
    from instance_scripts.optimize_pipeline import main, parse_args

    root, pred_dir = _make_majority_pred_root(tmp_path, 8)
    out = tmp_path / "out"

    main(
        parse_args(
            ["--root", str(root), "--pred-dir", str(pred_dir), "--n-images", "4",
             "--n-holdout", "4", "--n-trials", "2", "--workers", "1", "--output-dir", str(out),
             "--objective", "consensus"]
        )
    )  # fmt: skip

    written = json.loads((out / "summary.json").read_text())
    assert written["objective"] == "consensus"
    assert written["holdout"]["best"]["n_samples"] == 4, "one consensus sample per holdout scene"


@requires_map
def test_main_consensus_objective_needs_only_instance_masks(tmp_path: Path) -> None:
    import shutil

    from instance_scripts.optimize_pipeline import main, parse_args

    root, pred_dir = _make_majority_pred_root(tmp_path, 8)
    shutil.rmtree(root / "semantic_masks")
    out = tmp_path / "out"

    main(
        parse_args(
            ["--root", str(root), "--pred-dir", str(pred_dir), "--n-images", "4",
             "--n-holdout", "4", "--n-trials", "2", "--workers", "1", "--output-dir", str(out),
             "--objective", "consensus"]
        )
    )  # fmt: skip

    written = json.loads((out / "summary.json").read_text())
    assert written["holdout"]["best"]["n_samples"] == 4, "consensus contrails derived per scene"


@requires_map
def test_main_on_a_small_dataset_splits_half_and_half(tmp_path: Path) -> None:
    from instance_scripts.optimize_pipeline import main, parse_args

    root, pred_dir = _make_pred_root(tmp_path, 8)
    out = tmp_path / "out"

    summary = main(
        parse_args(
            ["--root", str(root), "--pred-dir", str(pred_dir), "--n-trials", "2",
             "--workers", "1", "--output-dir", str(out)]
        )
    )  # fmt: skip

    assert (summary["search_size"], summary["holdout_size"]) == (4, 4), (
        "no default of 1,000 scenes: half and half, never an empty holdout"
    )
