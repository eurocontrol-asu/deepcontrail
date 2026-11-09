"""Tests for deepcontrail.instance._evaluation (samples, splits, predictions, scoring)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig

if TYPE_CHECKING:
    from pathlib import Path

_S = 16
H = W = 64


requires_map = pytest.mark.skipif(
    importlib.util.find_spec("pycocotools") is None, reason="needs deepcontrail-instance[map]"
)


def _line_64(row: int, h: int = H, w: int = W) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[row, 8 : w - 8] = True
    return mask


def test_load_prediction_npy_returns_float32_probabilities(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import load_prediction

    probs = np.array([[0.2, 0.5], [0.8, 1.0]], dtype=np.float16)
    path = tmp_path / "s.npy"
    np.save(path, probs)

    out = load_prediction(path)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, probs.astype(np.float32))


def test_load_prediction_npy_rejects_out_of_range(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import load_prediction

    path = tmp_path / "s.npy"
    np.save(path, np.array([[0.0, 1.5]], dtype=np.float32))

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        load_prediction(path)


def test_load_prediction_npy_rejects_non_2d(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import load_prediction

    path = tmp_path / "s.npy"
    np.save(path, np.zeros((2, 3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="2-D"):
        load_prediction(path)


def test_load_prediction_png_0_255_and_0_1_become_0_1_floats(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import load_prediction

    arr = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(tmp_path / "a.png")
    Image.fromarray((arr > 0).astype(np.uint8), mode="L").save(tmp_path / "b.png")

    a = load_prediction(tmp_path / "a.png")
    b = load_prediction(tmp_path / "b.png")

    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, [[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(b, a)


def test_load_prediction_unknown_suffix_raises(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import load_prediction

    path = tmp_path / "s.tif"
    path.write_bytes(b"x")

    with pytest.raises(ValueError, match="suffix"):
        load_prediction(path)


def _line(row: int) -> np.ndarray:
    mask = np.zeros((_S, _S), dtype=bool)
    mask[row, 2:14] = True
    return mask


def _scene_arrays() -> tuple[np.ndarray, np.ndarray]:
    """3 annotator slots: slot 0 drew two contrails, slot 1 nothing, slot 2 one contrail."""
    inst = np.full((2, _S, _S, 3), -1, dtype=np.int8)
    inst[0, :, :, 0] = _line(3)
    inst[1, :, :, 0] = _line(9)
    inst[:, :, :, 1] = 0
    inst[0, :, :, 2] = _line(5)
    ann = np.zeros((_S, _S, 3), dtype=np.uint8)
    ann[..., 0] = _line(3) | _line(9)
    ann[..., 2] = _line(5)
    return ann, inst


def test_build_samples_one_per_annotator_slot_and_never_minus_one() -> None:
    from deepcontrail.instance._evaluation import build_samples

    ann, inst = _scene_arrays()

    samples = build_samples(inst, "s")

    assert [s.slot for s in samples] == [0, 1, 2]
    assert [len(s.gt) for s in samples] == [2, 0, 1], "-1 (not drawn) never becomes GT"
    assert np.array_equal(samples[0].annotator_mask, ann[..., 0]), (
        "an annotator's mask is the union of their contrails"
    )
    assert samples[0].annotator_mask.dtype == np.uint8


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
            line = _line_64(10 + 3 * (i % 12))
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


def test_select_annotator_samples_split_by_scene_and_non_empty(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import select_annotator_samples

    root = _make_annotator_root(tmp_path, 12)

    search, holdout = select_annotator_samples(root, 3, 3, seed=42)
    again = select_annotator_samples(root, 3, 3, seed=42)

    assert len(search) >= 3
    assert len(holdout) >= 3
    assert all(s.gt for s in search + holdout), "only annotators who drew something are samples"
    assert {s.stem for s in search}.isdisjoint({s.stem for s in holdout}), "scenes never split"
    assert [s.stem for s in search] == [s.stem for s in again[0]], "deterministic for a seed"


def test_select_annotator_samples_raises_when_not_enough(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import select_annotator_samples

    root = _make_annotator_root(tmp_path, 6)

    with pytest.raises(ValueError, match="annotator samples"):
        select_annotator_samples(root, 10, 10, seed=42)


# --- search space ---


@requires_map
def test_evaluate_annotator_masks_scores_pooled_map(tmp_path: Path) -> None:
    from deepcontrail.instance._evaluation import evaluate_annotator_masks, select_annotator_samples

    root = _make_annotator_root(tmp_path, 12)
    search, _ = select_annotator_samples(root, 3, 3, seed=42)

    r = evaluate_annotator_masks(search, MannsteinConfig(), workers=1)

    assert r["map"] == pytest.approx(1.0), "long separate lines are recovered exactly"
    assert r["n_samples"] == len(search)
    assert "ap_per_threshold" not in r, "the search objective skips the per-threshold APs"


def _line_probs(row: int) -> np.ndarray:
    """Probability map with one long horizontal contrail at ``row``."""
    probs = np.zeros((H, W), dtype=np.float32)
    probs[row, 8 : W - 8] = 0.9
    return probs


def _line_gt(row: int) -> np.ndarray:
    gt = np.zeros((H, W), dtype=bool)
    gt[row, 8 : W - 8] = True
    return gt


@requires_map
def test_evaluate_pipeline_scores_synthetic_lines() -> None:
    from deepcontrail.instance._evaluation import evaluate_pipeline

    probs = [_line_probs(20), _line_probs(40)]
    gts = [[[_line_gt(20)]], [[_line_gt(40)]]]
    config = InstancePipelineConfig(grow_steps=0)

    metrics = evaluate_pipeline(probs, gts, config, workers=1, match_iou=0.5)

    assert metrics["map"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0), "F1 stays available as a reference"
    assert metrics["mean_pred_instances"] == pytest.approx(1.0)
    assert "ap_per_threshold" not in metrics


@requires_map
def test_evaluate_pipeline_scores_each_annotator_as_a_sample() -> None:
    from deepcontrail.instance._evaluation import evaluate_pipeline

    probs = [_line_probs(20)]
    gts = [[[_line_gt(20)], [], [_line_gt(40)]]]

    metrics = evaluate_pipeline(
        probs, gts, InstancePipelineConfig(grow_steps=0), workers=1, match_iou=0.5
    )

    assert metrics["n_samples"] == 2, "the annotator who drew nothing is skipped"
    assert metrics["recall"] == pytest.approx(0.5)


@requires_map
def test_evaluate_pipeline_all_empty_returns_zero_map() -> None:
    from deepcontrail.instance._evaluation import evaluate_pipeline

    probs = [np.zeros((H, W), np.float32)]
    gts = [[[_line_gt(20)]]]

    metrics = evaluate_pipeline(probs, gts, InstancePipelineConfig(), workers=1, match_iou=0.5)

    assert metrics["map"] == 0.0
    assert metrics["f1"] == 0.0


@requires_map
def test_evaluate_pipeline_consensus_scores_once_per_scene() -> None:
    from deepcontrail.instance._evaluation import evaluate_pipeline

    probs = [_line_probs(20), _line_probs(30)]
    gts = [[[_line_gt(20)], [_line_gt(20)], []], [[_line_gt(30)], [], []]]
    consensus = [[_line_gt(20)], [_line_gt(30)]]
    config = InstancePipelineConfig(mannstein=MannsteinConfig(min_pixels=3, min_length=2.0))

    metrics = evaluate_pipeline(probs, gts, config, workers=1, match_iou=0.5, consensus=consensus)

    assert metrics["n_samples"] == 2, "one sample per scene against its consensus contrails"
    assert metrics["map"] == pytest.approx(1.0)


# --- scene splits (small datasets) ---


def test_split_scenes_defaults_to_half_and_half_and_is_reproducible():
    from deepcontrail.instance._evaluation import split_scenes

    stems = [f"s{i:02d}" for i in range(11)]

    search, holdout = split_scenes(stems, seed=3)
    again = split_scenes(stems, seed=3)

    assert (len(search), len(holdout)) == (5, 6)
    assert set(search).isdisjoint(holdout)
    assert (search, holdout) == again


@pytest.mark.parametrize(
    ("n", "n_search", "n_holdout"), [(500, 1000, None), (1, None, None), (10, 10, None), (10, 3, 0)]
)
def test_split_scenes_refuses_an_empty_holdout(n, n_search, n_holdout):
    from deepcontrail.instance._evaluation import split_scenes

    stems = [f"s{i}" for i in range(n)]

    with pytest.raises(ValueError, match=r"at least 1|need"):
        split_scenes(stems, seed=0, n_search=n_search, n_holdout=n_holdout)


# --- parallel detection without fork ---


def test_detect_all_runs_in_spawned_worker_processes():
    code = (
        "import multiprocessing, numpy as np\n"
        "from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig\n"
        "from deepcontrail.instance._evaluation import detect_all\n"
        "if __name__ == '__main__':\n"
        "    multiprocessing.set_start_method('spawn')\n"
        "    probs = []\n"
        "    for row in (10, 30, 50):\n"
        "        p = np.zeros((64, 64), np.float32); p[row, 8:56] = 0.9; probs.append(p)\n"
        "    cfg = InstancePipelineConfig(mannstein=MannsteinConfig(min_pixels=3, min_length=2.0))\n"
        "    out = detect_all(probs, cfg, workers=2)\n"
        "    print([len(masks) for masks, _ in out])\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[1, 1, 1]", "no reliance on fork-inherited globals"
