"""Tests for scripts.evaluate_scenes_instances (streamed whole-scene instance mAP)."""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

requires_map = pytest.mark.skipif(
    importlib.util.find_spec("pycocotools") is None, reason="needs deepcontrail-instance[map]"
)

_SIZE = 32


def _line(row: int, x0: int = 4, x1: int = 28) -> np.ndarray:
    """Create a one-pixel-tall horizontal line on an empty ``_SIZE x _SIZE`` grid."""
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[row, x0:x1] = True
    return mask


def _instance_array(lines: list[np.ndarray]) -> np.ndarray:
    """``(N, H, W, 1)`` int8 instance masks, one annotator, 1 inside each line."""
    arr = np.zeros((len(lines), _SIZE, _SIZE, 1), dtype=np.int8)
    for i, mask in enumerate(lines):
        arr[i, :, :, 0] = mask.astype(np.int8)
    return arr


def _pred_from_lines(lines: list[np.ndarray]) -> np.ndarray:
    """Create a probability map that is 1.0 exactly on the given lines, 0.0 elsewhere."""
    probs = np.zeros((_SIZE, _SIZE), dtype=np.float32)
    for mask in lines:
        probs[mask] = 1.0
    return probs


def _lenient_pipeline_yaml(tmp_path: Path) -> Path:
    """Create a pipeline that keeps any small straight line and does not grow (masks come back whole)."""
    from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig

    cfg = InstancePipelineConfig(
        semantic_threshold=0.5,
        grow_steps=0,
        probability_scores=False,
        mannstein=MannsteinConfig(min_pixels=3, min_length=2.0, min_straightness=0.0),
    )
    path = tmp_path / "pipeline.yaml"
    cfg.to_yaml(path)
    return path


def _two_lines() -> list[np.ndarray]:
    """Two disjoint horizontal lines, well separated so each is its own instance."""
    return [_line(8), _line(24)]


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """3 test scenes: s1 perfect (2 lines), s2 empty, s3 half (detects one of two)."""
    root = tmp_path / "data"
    pred_dir = tmp_path / "preds"
    (root / "instance_masks" / "test").mkdir(parents=True)
    pred_dir.mkdir()

    two = _two_lines()
    scenes = {
        "s1": {
            "split": "test",
            "group": "day1",
            "contrails": [{"confidence": "High"}, {"confidence": "Low"}],
        },
        "s2": {"split": "test", "group": "day1", "contrails": []},
        "s3": {
            "split": "test",
            "group": "day2",
            "contrails": [{"confidence": "High"}, {"confidence": "Medium"}],
        },
    }
    for stem, inst, pred in (
        ("s1", _instance_array(two), _pred_from_lines(two)),
        ("s2", np.zeros((0, _SIZE, _SIZE, 1), np.int8), np.zeros((_SIZE, _SIZE), np.float32)),
        ("s3", _instance_array(two), _pred_from_lines([two[0]])),
    ):
        np.save(root / "instance_masks" / "test" / f"{stem}.npy", inst)
        np.save(pred_dir / f"{stem}.npy", pred)
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / "scenes.json").write_text(json.dumps(scenes))
    return root, pred_dir


@requires_map
def test_main_streamed_scores_scenes_and_writes_json(tmp_path: Path) -> None:
    """A streamed run over three scenes writes the JSON with the expected structure."""
    from instance_scripts.evaluate_scenes_instances import main

    root, pred_dir = _write_dataset(tmp_path)
    out = tmp_path / "scores.json"

    code = main(
        [
            "--root",
            str(root),
            "--pred-dir",
            str(pred_dir),
            "--output",
            str(out),
            "--pipeline-config",
            str(_lenient_pipeline_yaml(tmp_path)),
            "--workers",
            "1",
        ]
    )

    assert code == 0
    assert out.is_file()
    result = json.loads(out.read_text())
    assert result["n_scenes"] == 3
    assert 0.0 <= result["map"] <= 1.0
    # JSON serializes float keys as strings
    assert set(result["ap_per_threshold"]) == {"0.25", "0.35", "0.45", "0.55", "0.65", "0.75"}
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert "with_empty" in result
    assert "bootstrap" in result
    assert result["bootstrap"]["n_groups"] == 2, "day1 (s1, s2) and day2 (s3)"


@requires_map
def test_perfect_scene_alone_reaches_map_one(tmp_path: Path) -> None:
    """Limiting to the perfect scene scores a clean map of 1.0."""
    from instance_scripts.evaluate_scenes_instances import main

    root, pred_dir = _write_dataset(tmp_path)
    out = tmp_path / "scores.json"

    main(
        [
            "--root",
            str(root),
            "--pred-dir",
            str(pred_dir),
            "--output",
            str(out),
            "--pipeline-config",
            str(_lenient_pipeline_yaml(tmp_path)),
            "--workers",
            "1",
            "--limit",
            "1",
        ]
    )

    result = json.loads(out.read_text())
    assert result["n_scenes"] == 1
    assert result["map"] == pytest.approx(1.0), "s1's two lines are recovered exactly"
    assert result["f1"] == pytest.approx(1.0)


@requires_map
def test_by_confidence_reports_every_level(tmp_path: Path) -> None:
    """--by-confidence adds a map and recall per confidence level."""
    from instance_scripts.evaluate_scenes_instances import main

    root, pred_dir = _write_dataset(tmp_path)
    out = tmp_path / "scores.json"

    main(
        [
            "--root",
            str(root),
            "--pred-dir",
            str(pred_dir),
            "--output",
            str(out),
            "--pipeline-config",
            str(_lenient_pipeline_yaml(tmp_path)),
            "--workers",
            "1",
            "--by-confidence",
        ]
    )

    result = json.loads(out.read_text())
    assert set(result["by_confidence"]) == {"High", "Medium", "Low"}
    for scores in result["by_confidence"].values():
        assert 0.0 <= scores["map"] <= 1.0
        assert 0.0 <= scores["recall"] <= 1.0


@requires_map
def test_by_confidence_without_scenes_json_errors(tmp_path: Path) -> None:
    """--by-confidence needs scenes.json for the confidences; a run without it fails clearly."""
    from instance_scripts.evaluate_scenes_instances import main

    root, pred_dir = _write_dataset(tmp_path)
    (root / "metadata" / "scenes.json").unlink()
    out = tmp_path / "scores.json"

    with pytest.raises(FileNotFoundError, match=r"scenes\.json"):
        main(
            [
                "--root",
                str(root),
                "--pred-dir",
                str(pred_dir),
                "--output",
                str(out),
                "--pipeline-config",
                str(_lenient_pipeline_yaml(tmp_path)),
                "--workers",
                "1",
                "--by-confidence",
            ]
        )


def test_parse_args_rejects_bundle_with_explicit_config() -> None:
    """--bundle already supplies the config and score model; combining them is an error."""
    from instance_scripts.evaluate_scenes_instances import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--root",
                "/nonexistent",
                "--pred-dir",
                "/nonexistent",
                "--output",
                "/nonexistent/scores.json",
                "--bundle",
                "b",
                "--pipeline-config",
                "p.yaml",
            ]
        )
