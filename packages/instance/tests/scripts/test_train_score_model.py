"""Tests for scripts.train_score_model (fit a ScoreModel on validation predictions)."""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig, ScoreModel

if TYPE_CHECKING:
    from pathlib import Path

H = W = 64


requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="needs deepcontrail-instance[scores]"
)


def _line(row: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[row, 8 : W - 8] = True
    return mask


def _make_pred_root(tmp_path: Path, n: int) -> tuple[Path, Path]:
    """Train-split GT (2 of 3 annotators drew the line) plus a folder of predictions."""
    inst_dir = tmp_path / "root" / "instance_masks" / "train"
    sem_dir = tmp_path / "root" / "semantic_masks" / "train"
    pred_dir = tmp_path / "preds"
    for d in (inst_dir, sem_dir, pred_dir):
        d.mkdir(parents=True)
    for i in range(n):
        stem = f"scene_{i:03d}"
        row = 10 + 4 * (i % 10)
        probs = _line(row).astype(np.float32) * (0.6 + 0.03 * i)
        np.save(pred_dir / f"{stem}.npy", probs.astype(np.float16))
        raw = np.full((1, H, W, 3), -1, dtype=np.int8)
        raw[0, :, :, 0] = _line(row)
        raw[0, :, :, 1] = _line(row)
        np.save(inst_dir / f"{stem}.npy", raw)
        Image.fromarray(_line(row).astype(np.uint8) * 255, mode="L").save(sem_dir / f"{stem}.png")
    return tmp_path / "root", pred_dir


@requires_sklearn
def test_main_writes_a_loadable_score_model_and_holdout_summary(tmp_path: Path) -> None:
    from instance_scripts.train_score_model import main, parse_args

    root, pred_dir = _make_pred_root(tmp_path, 12)
    config = InstancePipelineConfig(mannstein=MannsteinConfig(min_pixels=3, min_length=2.0))
    config_path = tmp_path / "pipeline.yaml"
    config.to_yaml(config_path)
    out = tmp_path / "out"

    summary = main(
        parse_args(
            ["--root", str(root), "--pred-dir", str(pred_dir), "--pipeline-config",
             str(config_path), "--n-images", "6", "--n-holdout", "6", "--workers", "1",
             "--output-dir", str(out)]
        )
    )  # fmt: skip

    model = ScoreModel.from_json(out / "score_model.json")
    assert model.pipeline_config == config, "the model remembers the config it was trained with"
    written = json.loads((out / "summary.json").read_text())
    assert set(written["holdout"]) == {"pipeline_scores", "learned_scores"}
    for scores in written["holdout"].values():
        assert set(scores) >= {"annotators_map", "consensus_map"}
    assert summary["n_training_contrails"] > 0


@requires_sklearn
def test_main_needs_only_instance_masks(tmp_path: Path) -> None:
    import shutil

    from instance_scripts.train_score_model import main, parse_args

    root, pred_dir = _make_pred_root(tmp_path, 12)
    shutil.rmtree(root / "semantic_masks")
    out = tmp_path / "out"

    main(
        parse_args(
            ["--root", str(root), "--pred-dir", str(pred_dir), "--n-images", "6",
             "--n-holdout", "6", "--workers", "1", "--output-dir", str(out)]
        )
    )  # fmt: skip

    assert (out / "score_model.json").exists()


@requires_sklearn
def test_main_on_a_small_dataset_splits_half_and_half(tmp_path: Path) -> None:
    from instance_scripts.train_score_model import main, parse_args

    root, pred_dir = _make_pred_root(tmp_path, 12)
    out = tmp_path / "out"

    summary = main(
        parse_args(
            ["--root", str(root), "--pred-dir", str(pred_dir), "--workers", "1",
             "--output-dir", str(out)]
        )
    )  # fmt: skip

    assert (summary["n_training_scenes"], summary["n_holdout_scenes"]) == (6, 6)
