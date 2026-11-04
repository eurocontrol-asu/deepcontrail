"""Tests for deepcontrail.instance.bundle (the instance step stored in a pretrained bundle)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from deepcontrail.instance import (
    InstancePipelineConfig,
    InstanceStep,
    MannsteinConfig,
    ScoreModel,
    detect_instances_from_probabilities,
    load_instance_step,
    save_instance_step,
)

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG = InstancePipelineConfig(
    grow_steps=0, mannstein=MannsteinConfig(min_pixels=3, min_length=2.0)
)


def _score_model(**trained_on: object) -> ScoreModel:
    """Build a tree-less model (constant score): enough for the bundle, no scikit-learn."""
    return ScoreModel(0.25, (), _CONFIG, trained_on=dict(trained_on))


def _probs(size: int = 32) -> np.ndarray:
    probs = np.zeros((size, size), dtype=np.float32)
    probs[8, 4:28] = 0.9
    probs[20, 4:28] = 0.8
    return probs


def test_save_and_load_instance_step_round_trip(tmp_path: Path):
    model = _score_model(tta=True, scene_shape=[32, 32])

    save_instance_step(tmp_path, _CONFIG, model)
    step = load_instance_step(tmp_path, tta=True)

    assert step.config == _CONFIG
    assert step.score_model is not None
    assert step.score_model.trained_on["tta"] is True
    assert (tmp_path / "instance_pipeline.yaml").is_file()
    assert (tmp_path / "score_model.json").is_file()


def test_instance_step_runs_the_pipeline_with_its_score_model(tmp_path: Path):
    model = _score_model(scene_shape=[32, 32])
    save_instance_step(tmp_path, _CONFIG, model)

    masks, scores = load_instance_step(tmp_path)(_probs())

    expected_masks, expected_scores = detect_instances_from_probabilities(_probs(), _CONFIG, model)
    assert len(masks) == len(expected_masks) == 2
    assert all(np.array_equal(a, b) for a, b in zip(masks, expected_masks, strict=True))
    assert scores == expected_scores == [0.25, 0.25], "the learned score replaces the pipeline's"


def test_load_instance_step_without_instance_files_uses_the_defaults(tmp_path: Path):
    step = load_instance_step(tmp_path)

    assert step == InstanceStep(InstancePipelineConfig(), None), (
        "a bundle without the instance files (e.g. v0.1.0) runs the default pipeline"
    )


def test_save_instance_step_without_score_model_writes_only_the_pipeline(tmp_path: Path):
    save_instance_step(tmp_path, _CONFIG)

    assert (tmp_path / "instance_pipeline.yaml").is_file()
    assert not (tmp_path / "score_model.json").exists()
    assert load_instance_step(tmp_path).score_model is None


@pytest.mark.parametrize(("trained_with", "used_with"), [(True, False), (False, True)])
def test_load_instance_step_rejects_a_score_model_made_with_other_tta(
    tmp_path: Path, trained_with: bool, used_with: bool
):
    save_instance_step(tmp_path, _CONFIG, _score_model(tta=trained_with))

    with pytest.raises(ValueError, match="TTA"):
        load_instance_step(tmp_path, tta=used_with)


def test_load_instance_step_rejects_a_pipeline_file_that_disagrees_with_the_score_model(
    tmp_path: Path,
):
    save_instance_step(tmp_path, _CONFIG, _score_model())
    InstancePipelineConfig().to_yaml(tmp_path / "instance_pipeline.yaml")

    with pytest.raises(ValueError, match="pipeline config"):
        load_instance_step(tmp_path)
