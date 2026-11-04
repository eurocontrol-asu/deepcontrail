"""Tests for scripts.evaluate_instances (prediction loading and pipeline selection)."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from PIL import Image

from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig

if TYPE_CHECKING:
    from pathlib import Path

_BASE_ARGS = ["--root", "/nonexistent", "--use-gt-semantic"]


requires_map = pytest.mark.skipif(
    importlib.util.find_spec("pycocotools") is None, reason="needs deepcontrail-instance[map]"
)
requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="needs deepcontrail-instance[scores]"
)


def test_build_pipeline_config_baseline_is_plain_mannstein() -> None:
    from instance_scripts.evaluate_instances import build_pipeline_config, parse_args

    cfg = build_pipeline_config(parse_args(_BASE_ARGS))

    assert cfg.semantic_threshold == 0.5
    assert cfg.grow_steps == 0, "the baseline does not grow instances"
    assert not cfg.probability_scores, "the baseline keeps Mannstein's own scores"
    assert cfg.mannstein == MannsteinConfig()


def test_build_pipeline_config_baseline_uses_threshold_and_mannstein_yaml(tmp_path: Path) -> None:
    from instance_scripts.evaluate_instances import build_pipeline_config, parse_args

    path = tmp_path / "m.yaml"
    path.write_text(yaml.dump({"mannstein": {"num_kernels": 30, "kernel_size": 9}}))

    cfg = build_pipeline_config(
        parse_args([*_BASE_ARGS, "--threshold", "0.42", "--mannstein-config", str(path)])
    )

    assert cfg.semantic_threshold == 0.42
    assert cfg.mannstein.num_kernels == 30


def test_build_pipeline_config_loads_pipeline_yaml(tmp_path: Path) -> None:
    from instance_scripts.evaluate_instances import build_pipeline_config, parse_args

    expected = InstancePipelineConfig(semantic_threshold=0.45, grow_steps=2)
    path = tmp_path / "p.yaml"
    expected.to_yaml(path)

    cfg = build_pipeline_config(parse_args([*_BASE_ARGS, "--pipeline-config", str(path)]))

    assert cfg == expected


@pytest.mark.parametrize("extra", [["--threshold", "0.4"], ["--mannstein-config", "m.yaml"]])
def test_parse_args_rejects_pipeline_config_with_baseline_options(extra) -> None:
    from instance_scripts.evaluate_instances import parse_args

    with pytest.raises(SystemExit):
        parse_args([*_BASE_ARGS, "--pipeline-config", "p.yaml", *extra])


# --- annotator samples ---

_S = 16


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


def _write_root(tmp_path: Path) -> Path:
    """Write a one-scene test split whose consensus equals annotator slot 0's mask."""
    ann, inst = _scene_arrays()
    for sub in ("annotator_masks", "instance_masks", "semantic_masks"):
        (tmp_path / sub / "test").mkdir(parents=True)
    np.save(tmp_path / "annotator_masks/test/s1.npy", ann)
    np.save(tmp_path / "instance_masks/test/s1.npy", inst)
    Image.fromarray((ann[..., 0] * 255).astype(np.uint8), mode="L").save(
        tmp_path / "semantic_masks/test/s1.png"
    )
    return tmp_path


@requires_map
@pytest.mark.parametrize("mode", [["--use-annotator-semantic"], ["--use-gt-semantic"]])
def test_main_scores_annotator_samples_and_skips_empty(tmp_path: Path, mode) -> None:
    from instance_scripts.evaluate_instances import main, parse_args

    root = _write_root(tmp_path)
    cfg = MannsteinConfig(min_pixels=3, min_length=2.0)
    cfg_path = tmp_path / "m.yaml"
    cfg_path.write_text(yaml.dump({"mannstein": cfg.model_dump()}))

    r = main(
        parse_args(
            ["--root", str(root), *mode, "--mannstein-config", str(cfg_path), "--workers", "1"]
        )
    )

    assert r["n_annotator_samples"] == 3
    assert r["n_samples"] == 2, "the annotator who drew nothing is skipped"
    assert r["n_skipped_empty"] == 1
    assert 0.0 <= r["map"] <= 1.0
    assert set(r["ap_per_threshold"]) == {0.25, 0.35, 0.45, 0.55, 0.65, 0.75}


@requires_map
def test_main_annotator_mode_is_perfect_on_own_masks(tmp_path: Path) -> None:
    from instance_scripts.evaluate_instances import main, parse_args

    root = _write_root(tmp_path)
    cfg_path = tmp_path / "m.yaml"
    cfg_path.write_text(
        yaml.dump({"mannstein": MannsteinConfig(min_pixels=3, min_length=2.0).model_dump()})
    )

    r = main(
        parse_args(
            [
                "--root",
                str(root),
                "--use-annotator-semantic",
                "--mannstein-config",
                str(cfg_path),
                "--workers",
                "1",
            ]
        )
    )

    assert r["map"] == pytest.approx(1.0), (
        "separate straight lines are recovered exactly from each annotator's mask"
    )
    assert r["f1"] == pytest.approx(1.0)


# --- consensus instance GT ---


def _write_majority_root(tmp_path: Path) -> Path:
    """One scene, 3 annotators: two drew the line at row 4 (a majority), one drew row 11 only."""
    inst = np.full((2, _S, _S, 3), -1, dtype=np.int8)
    inst[0, :, :, 0] = _line(4)
    inst[0, :, :, 1] = _line(4)
    inst[0, :, :, 2] = _line(11)
    ann = np.stack([_line(4), _line(4), _line(11)], axis=-1).astype(np.uint8)
    for sub in ("annotator_masks", "instance_masks", "semantic_masks"):
        (tmp_path / sub / "test").mkdir(parents=True)
    np.save(tmp_path / "annotator_masks/test/s1.npy", ann)
    np.save(tmp_path / "instance_masks/test/s1.npy", inst)
    Image.fromarray(_line(4).astype(np.uint8) * 255, mode="L").save(
        tmp_path / "semantic_masks/test/s1.png"
    )
    return tmp_path


def _lenient_config(tmp_path: Path) -> Path:
    path = tmp_path / "m.yaml"
    path.write_text(
        yaml.dump({"mannstein": MannsteinConfig(min_pixels=3, min_length=2.0).model_dump()})
    )
    return path


@requires_map
def test_consensus_gt_scores_against_the_majority_contrails(tmp_path: Path) -> None:
    from instance_scripts.evaluate_instances import main, parse_args

    root = _write_majority_root(tmp_path / "data")
    cfg = _lenient_config(tmp_path)

    r = main(
        parse_args(
            [
                "--root",
                str(root),
                "--use-gt-semantic",
                "--consensus-gt",
                "--mannstein-config",
                str(cfg),
                "--workers",
                "1",
            ]
        )
    )

    assert r["consensus"]["n_instances"] == 1, "only the contrail drawn by 2 of 3 annotators"
    assert r["consensus"]["map"] == pytest.approx(1.0)
    assert set(r["consensus"]["ap_per_threshold"]) == {0.25, 0.35, 0.45, 0.55, 0.65, 0.75}
    assert r["n_samples"] == 3, "the official per-annotator metric is still reported"


@requires_map
def test_consensus_gt_absent_unless_requested(tmp_path: Path) -> None:
    from instance_scripts.evaluate_instances import main, parse_args

    root = _write_majority_root(tmp_path / "data")

    r = main(
        parse_args(
            [
                "--root",
                str(root),
                "--use-gt-semantic",
                "--mannstein-config",
                str(_lenient_config(tmp_path)),
                "--workers",
                "1",
            ]
        )
    )

    assert "consensus" not in r


def test_consensus_gt_rejected_with_annotator_inputs() -> None:
    from instance_scripts.evaluate_instances import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--root", "/nonexistent", "--use-annotator-semantic", "--consensus-gt"])


# --- probability input ---


def _score_model_file(tmp_path: Path, config: InstancePipelineConfig) -> Path:
    from deepcontrail.instance import FEATURES, ScoreModel

    rng = np.random.default_rng(0)
    x = rng.random((200, len(FEATURES)))
    path = tmp_path / "score_model.json"
    ScoreModel.fit(x, x[:, 0], config).to_json(path)
    return path


@requires_sklearn
def test_main_with_score_model_rescores_and_records_it(tmp_path: Path) -> None:
    from instance_scripts.evaluate_instances import main, parse_args

    root = _write_majority_root(tmp_path / "data")
    config = InstancePipelineConfig(mannstein=MannsteinConfig(min_pixels=3, min_length=2.0))
    config_path = tmp_path / "p.yaml"
    config.to_yaml(config_path)
    model_path = _score_model_file(tmp_path, config)
    base = ["--root", str(root), "--use-gt-semantic", "--pipeline-config", str(config_path),
            "--workers", "1"]  # fmt: skip

    plain = main(parse_args(base))
    rescored = main(parse_args([*base, "--score-model", str(model_path)]))

    assert rescored["score_model"] == str(model_path)
    assert rescored["f1"] == plain["f1"], "a score model changes the ranking, not the masks"


def test_parse_args_rejects_score_model_without_one_map_per_scene() -> None:
    from instance_scripts.evaluate_instances import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            ["--root", "/nonexistent", "--use-annotator-semantic", "--score-model", "m.json"]
        )


@requires_map
def test_main_include_empty_reports_real_conditions_next_to_the_official_metric(
    tmp_path: Path,
) -> None:
    from instance_scripts.evaluate_instances import main, parse_args

    root = _write_root(tmp_path / "data")
    ann, _ = _scene_arrays()
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    np.save(pred_dir / "s1.npy", ann[..., 0].astype(np.float32) * 0.9)
    cfg = _lenient_config(tmp_path)
    base = ["--root", str(root), "--pred-dir", str(pred_dir), "--mannstein-config", str(cfg),
            "--workers", "1", "--consensus-gt"]  # fmt: skip

    official = main(parse_args(base))
    r = main(parse_args([*base, "--include-empty"]))

    assert r["map"] == official["map"], "the official numbers are unchanged"
    real = r["with_empty"]
    assert real["n_samples"] == 3, "the annotator who drew nothing is kept"
    assert real["fp"] > official["fp"], "detections on that annotator's sample are false"
    assert "with_empty" in r["consensus"]
    assert "with_empty" not in official


@requires_map
def test_main_needs_only_instance_masks_even_with_a_single_annotator(tmp_path: Path) -> None:
    from instance_scripts.evaluate_instances import main, parse_args

    root = tmp_path / "data"
    (root / "instance_masks" / "test").mkdir(parents=True)
    for i, rows in enumerate(([3, 9], [5], [])):
        inst = np.zeros((len(rows), _S, _S, 1), dtype=np.int8)
        for n, row in enumerate(rows):
            inst[n, :, :, 0] = _line(row)
        np.save(root / f"instance_masks/test/s{i}.npy", inst)
    preds = tmp_path / "preds"
    preds.mkdir()
    for i, rows in enumerate(([3, 9], [5], [])):
        probs = np.zeros((_S, _S), dtype=np.float32)
        for row in rows:
            probs[_line(row)] = 0.9
        np.save(preds / f"s{i}.npy", probs)
    cfg = _lenient_config(tmp_path)
    base = ["--root", str(root), "--mannstein-config", str(cfg), "--workers", "1"]

    own = main(parse_args([*base, "--use-annotator-semantic"]))
    ceiling = main(parse_args([*base, "--use-gt-semantic", "--consensus-gt", "--include-empty"]))
    end_to_end = main(parse_args([*base, "--pred-dir", str(preds), "--consensus-gt"]))

    assert own["n_annotator_samples"] == 3
    assert own["map"] == pytest.approx(1.0)
    assert ceiling["map"] == pytest.approx(1.0)
    assert ceiling["consensus"]["map"] == pytest.approx(ceiling["map"]), (
        "with one annotator, the consensus is that annotator"
    )
    assert end_to_end["map"] == pytest.approx(1.0)
