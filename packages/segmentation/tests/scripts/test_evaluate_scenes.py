"""Tests for scripts/evaluate_scenes.py — probability maps + labels -> pooled scores."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_scenes.py"
SIZE = 4


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_scenes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_metadata(root: Path) -> None:
    """Minimal dataset metadata (band manifest + dataset info) so the masks can be read."""
    bands = {
        "format_version": 1,
        "sensor": "synthetic",
        "dtype": "float32",
        "channels": [
            {"index": i, "key": key, "units": "K"}
            for i, key in enumerate(("ir_87", "ir_105", "ir_123"))
        ],
    }
    (root / "spectral_bands").mkdir(parents=True, exist_ok=True)
    (root / "spectral_bands" / "bands.json").write_text(json.dumps(bands))
    info = {
        "format_version": 1,
        "source": "synthetic",
        "records": {"train": 0, "test": 0},
        "labels": ["semantic", "instance"],
        "recommended_pixel_shift": 0.0,
    }
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "dataset_info.json").write_text(json.dumps(info))


def _write_semantic(root: Path, scene_id: str, split: str, mask: np.ndarray) -> None:
    """Write a {0,255} consensus mask PNG at ``semantic_masks/<split>/<id>.png``."""
    path = root / "semantic_masks" / split / f"{scene_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def _write_instances(root: Path, scene_id: str, split: str, instances: list[np.ndarray]) -> None:
    """Write an ``instance_masks/<split>/<id>.npy`` array ``(N, H, W, 1)`` int8 per instance."""
    array = np.zeros((len(instances), SIZE, SIZE, 1), np.int8)
    for i, mask in enumerate(instances):
        array[i, :, :, 0] = mask.astype(np.int8)
    path = root / "instance_masks" / split / f"{scene_id}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _write_pred(pred_dir: Path, scene_id: str, pred: np.ndarray) -> None:
    """Write a float probability map at ``pred_dir/<id>.npy``."""
    path = pred_dir / f"{scene_id}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pred)


def _box(rows: range, cols: range) -> np.ndarray:
    """Build a boolean mask True on ``rows x cols`` of an empty ``SIZE x SIZE`` grid."""
    mask = np.zeros((SIZE, SIZE), bool)
    for r in rows:
        for c in cols:
            mask[r, c] = True
    return mask


def _two_scene_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a dataset of two test scenes (A, B) with confidences, groups, projects and maps."""
    root = tmp_path / "data"
    pred_dir = tmp_path / "preds"
    root.mkdir()
    pred_dir.mkdir()
    _write_metadata(root)

    gt_a = _box(range(2), range(2))  # top-left 2x2
    gt_b = _box(range(2), range(2, 4))  # top-right 2x2
    high_a, low_a = _box(range(1), range(2)), _box(range(1, 2), range(2))
    med_b, low_b = _box(range(1), range(2, 4)), _box(range(1, 2), range(2, 4))

    pred_a = np.full((SIZE, SIZE), 0.1, np.float32)
    pred_a[gt_a] = 0.9
    pred_a[2, 0] = 0.9  # a spurious prediction on the background
    pred_b = np.full((SIZE, SIZE), 0.1, np.float32)
    pred_b[0, 2] = pred_b[0, 3] = pred_b[1, 2] = 0.9
    pred_b[1, 3] = 0.3  # a missed pixel (in gt, below the threshold)

    for scene_id, gt in (("A", gt_a), ("B", gt_b)):
        _write_semantic(root, scene_id, "test", gt)
    _write_instances(root, "A", "test", [high_a, low_a])
    _write_instances(root, "B", "test", [med_b, low_b])
    _write_pred(pred_dir, "A", pred_a)
    _write_pred(pred_dir, "B", pred_b)

    scenes = {
        "A": {
            "split": "test",
            "group": "day1",
            "project": "projA",
            "contrails": [{"key": "a1", "confidence": "High"}, {"key": "a2", "confidence": "Low"}],
        },
        "B": {
            "split": "test",
            "group": "day2",
            "project": "projB",
            "contrails": [
                {"key": "b1", "confidence": "Medium"},
                {"key": "b2", "confidence": "Low"},
            ],
        },
    }
    (root / "metadata" / "scenes.json").write_text(json.dumps(scenes))
    return root, pred_dir, tmp_path / "scores.json"


def test_pooled_dice_matches_a_hand_computation(tmp_path: Path):
    """Pooled Dice at 0.5 over the two scenes is 2*7 / (8 + 8) = 0.875."""
    root, pred_dir, out = _two_scene_dataset(tmp_path)

    code = _load_script().main(
        ["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out), "--threshold", "0.5"]
    )

    assert code == 0
    pooled = json.loads(out.read_text())["pooled"]
    assert pooled["dice_at_0.5"] == pytest.approx(0.875)
    assert pooled["dice_at_threshold"] == pytest.approx(0.875)


def test_per_confidence_level_recall_and_dice_match_hand_computations(tmp_path: Path):
    """recall_L = |pred & gt_L| / |gt_L|; dice_L ignores predictions on other levels."""
    root, pred_dir, out = _two_scene_dataset(tmp_path)

    _load_script().main(
        ["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out), "--threshold", "0.5"]
    )

    level = json.loads(out.read_text())["by_confidence"]
    # High appears only in scene A: predicted exactly, one spurious background pixel -> dice 4/5.
    assert level["High"]["recall"] == pytest.approx(1.0)
    assert level["High"]["dice"] == pytest.approx(0.8)
    # Medium appears only in scene B; the scene-A background pixel is a false positive for it.
    assert level["Medium"]["recall"] == pytest.approx(1.0)
    assert level["Medium"]["dice"] == pytest.approx(0.8)
    # Low is in both scenes; scene B misses one Low pixel -> recall 3/4, dice 6/8.
    assert level["Low"]["recall"] == pytest.approx(0.75)
    assert level["Low"]["dice"] == pytest.approx(0.75)


def test_dice_without_low_ignores_low_pixels_in_both_sides(tmp_path: Path):
    """Removing the Low pixels from pred and gt gives pooled Dice 2*4 / (5 + 4) = 8/9."""
    root, pred_dir, out = _two_scene_dataset(tmp_path)

    _load_script().main(
        ["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out), "--threshold", "0.5"]
    )

    dice = json.loads(out.read_text())["pooled"]["dice_without_low"]
    assert dice == pytest.approx(8 / 9)


def test_best_threshold_of_a_map_perfect_above_0_7_is_at_least_0_7(tmp_path: Path):
    """A map whose best mask sits at 0.72/0.95 is scored best at a threshold >= 0.7."""
    root = tmp_path / "data"
    pred_dir = tmp_path / "preds"
    root.mkdir()
    pred_dir.mkdir()
    _write_metadata(root)
    gt = _box(range(1), range(2))  # two pixels
    pred = np.zeros((SIZE, SIZE), np.float32)
    pred[gt] = 0.95
    pred[0, 2] = pred[0, 3] = pred[1, 0] = 0.72  # background, just above 0.70
    _write_semantic(root, "S", "test", gt)
    _write_pred(pred_dir, "S", pred)
    out = tmp_path / "scores.json"
    (root / "metadata" / "scenes.json").write_text(
        json.dumps({"S": {"split": "test", "group": "g1", "project": "p1", "contrails": []}})
    )

    _load_script().main(["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out)])

    pooled = json.loads(out.read_text())["pooled"]
    assert pooled["best_threshold"] >= 0.7, "the best mask is only reachable at high thresholds"
    assert pooled["best_dice"] == pytest.approx(1.0)


def test_bootstrap_interval_contains_the_point_estimate(tmp_path: Path):
    """The day-bootstrap Dice interval brackets the pooled point estimate."""
    root, pred_dir, out = _two_scene_dataset(tmp_path)

    _load_script().main(
        [
            "--root",
            str(root),
            "--pred-dir",
            str(pred_dir),
            "--out",
            str(out),
            "--threshold",
            "0.5",
            "--bootstrap",
            "300",
            "--seed",
            "0",
        ]
    )

    result = json.loads(out.read_text())
    point = result["pooled"]["dice_at_threshold"]
    lo, hi = result["bootstrap"]["dice"]
    assert lo <= point <= hi, f"{point} must lie inside the bootstrap interval [{lo}, {hi}]"


def test_per_project_dice_and_scene_counts(tmp_path: Path):
    """Each project reports its pooled Dice and number of scenes."""
    root, pred_dir, out = _two_scene_dataset(tmp_path)

    _load_script().main(
        ["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out), "--threshold", "0.5"]
    )

    by_project = json.loads(out.read_text())["by_project"]
    # Scene A alone: 2*4 / (5 + 4) = 8/9; scene B alone: 2*3 / (3 + 4) = 6/7.
    assert by_project["projA"]["dice"] == pytest.approx(8 / 9)
    assert by_project["projA"]["n_scenes"] == 1
    assert by_project["projB"]["dice"] == pytest.approx(6 / 7)
    assert by_project["projB"]["n_scenes"] == 1


def test_null_confidence_contrails_are_counted_only(tmp_path: Path):
    """A null-confidence instance is its own 'None' level, reported only as a count."""
    root = tmp_path / "data"
    pred_dir = tmp_path / "preds"
    root.mkdir()
    pred_dir.mkdir()
    _write_metadata(root)
    gt = _box(range(1), range(1))  # one pixel
    high = _box(range(1), range(1))
    unconfirmed = _box(range(1, 2), range(1, 2))  # a separate pixel, no confidence
    pred = np.full((SIZE, SIZE), 0.1, np.float32)
    pred[gt] = 0.9
    _write_semantic(root, "N", "test", gt)
    _write_instances(root, "N", "test", [high, unconfirmed])
    _write_pred(pred_dir, "N", pred)
    (root / "metadata" / "scenes.json").write_text(
        json.dumps(
            {
                "N": {
                    "split": "test",
                    "group": "g",
                    "project": "p",
                    "contrails": [
                        {"key": "c1", "confidence": "High"},
                        {"key": "c2", "confidence": None},
                    ],
                }
            }
        )
    )
    out = tmp_path / "scores.json"

    _load_script().main(["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out)])

    level = json.loads(out.read_text())["by_confidence"]
    assert level["None"]["n_instances"] == 1
    assert level["High"]["dice"] == pytest.approx(1.0), "the null instance is not scored as High"


def test_split_dir_is_found_in_train_without_scenes_meta(tmp_path: Path):
    """Without scenes.json, a mask is found under train/ and the scene is still scored."""
    root = tmp_path / "data"
    pred_dir = tmp_path / "preds"
    root.mkdir()
    pred_dir.mkdir()
    _write_metadata(root)
    gt = _box(range(1), range(2))
    pred = np.full((SIZE, SIZE), 0.1, np.float32)
    pred[gt] = 0.9
    _write_semantic(root, "T", "train", gt)
    _write_pred(pred_dir, "T", pred)
    out = tmp_path / "scores.json"

    code = _load_script().main(
        ["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out), "--threshold", "0.5"]
    )

    assert code == 0
    result = json.loads(out.read_text())
    assert result["config"]["n_scenes"] == 1
    assert result["pooled"]["dice_at_threshold"] == pytest.approx(1.0)
    assert "by_confidence" not in result, "no scenes.json -> no per-level section"


def test_a_prediction_absent_from_scenes_meta_is_still_scored(tmp_path: Path):
    """A map whose scene is not in scenes.json is found by fallback and scored on its own."""
    root, pred_dir, out = _two_scene_dataset(tmp_path)
    gt = _box(range(1), range(1))
    pred = np.full((SIZE, SIZE), 0.1, np.float32)
    pred[gt] = 0.9
    _write_semantic(root, "Z", "test", gt)  # a mask under test/, but no scenes.json entry
    _write_pred(pred_dir, "Z", pred)

    code = _load_script().main(
        ["--root", str(root), "--pred-dir", str(pred_dir), "--out", str(out), "--threshold", "0.5"]
    )

    assert code == 0, "an unknown scene must not crash the run"
    result = json.loads(out.read_text())
    assert result["config"]["n_scenes"] == 3
    # pooled overlap 8, predicted 9, target 9 -> 2*8 / (9 + 9) = 8/9
    assert result["pooled"]["dice_at_threshold"] == pytest.approx(8 / 9)
