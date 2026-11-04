"""Tests for deepcontrail.instance.scoring (learned contrail scores)."""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from deepcontrail.instance import (
    FEATURES,
    InstancePipelineConfig,
    MannsteinConfig,
    ScoreModel,
    detect_instances_from_probabilities,
    instance_features,
    match_quality,
    training_data,
)

if TYPE_CHECKING:
    from pathlib import Path

_S = 64


def _line(row: int, start: int = 8, stop: int = 56, width: int = 1) -> np.ndarray:
    mask = np.zeros((_S, _S), dtype=bool)
    mask[row : row + width, start:stop] = True
    return mask


def _training_set(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.random((n, len(FEATURES)))
    mean_prob = FEATURES.index("mean_prob")
    return x, x[:, mean_prob] ** 2


# Fitting needs the [scores] extra (scikit-learn); predicting and the JSON format do not.
requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="needs deepcontrail-instance[scores]"
)

_CONFIG = InstancePipelineConfig(
    grow_steps=0, mannstein=MannsteinConfig(min_pixels=3, min_length=2.0)
)


# --- features and labels ---


def test_instance_features_describe_size_shape_and_probability() -> None:
    mask = _line(10, width=2)
    probs = np.where(mask, 0.8, 0.1).astype(np.float32)

    feats = instance_features(mask, probs, score=0.6, scene_masks=[mask])

    named = dict(zip(FEATURES, feats, strict=True))
    assert named["n_pixels"] == 96
    assert named["length"] == pytest.approx(48.0, rel=0.05)
    assert named["width"] == pytest.approx(2.0, rel=0.05)
    assert named["mean_prob"] == pytest.approx(0.8)
    assert named["ring_mean_prob"] == pytest.approx(0.1), "pixels just around the contrail"
    assert named["pipeline_score"] == pytest.approx(0.6)
    assert named["n_scene_instances"] == 1


def test_match_quality_is_mean_best_iou_over_annotators_who_drew() -> None:
    label = match_quality(_line(10), [[_line(10)], [_line(20)], []])

    assert label == pytest.approx(0.5), "IoU 1 with the first annotator, 0 with the second"


def test_match_quality_without_annotations_is_zero() -> None:
    assert match_quality(_line(10), [[], []]) == 0.0


def test_training_data_has_one_row_per_detected_contrail() -> None:
    probs = (_line(10) | _line(30)).astype(np.float32) * 0.9
    gts = [[_line(10), _line(30)], [_line(10)]]

    x, y = training_data([probs], [gts], _CONFIG)

    masks, _ = detect_instances_from_probabilities(probs, _CONFIG)
    assert x.shape == (len(masks), len(FEATURES))
    assert y.shape == (len(masks),)
    assert np.all((y >= 0.0) & (y <= 1.0))


# --- the model ---


@requires_sklearn
def test_score_model_matches_scikit_learn_exactly() -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    x, y = _training_set()
    reference = HistGradientBoostingRegressor(**ScoreModel.ESTIMATOR_SETTINGS).fit(x, y)

    model = ScoreModel.fit(x, y, _CONFIG)

    np.testing.assert_allclose(
        model.predict(x), np.clip(reference.predict(x), 0.0, 1.0), rtol=0, atol=1e-12
    )


@requires_sklearn
def test_score_model_learns_what_predicts_a_good_match() -> None:
    x, y = _training_set()
    model = ScoreModel.fit(x, y, _CONFIG)
    low, high = x[:1].copy(), x[:1].copy()
    mean_prob = FEATURES.index("mean_prob")
    low[0, mean_prob], high[0, mean_prob] = 0.1, 0.9

    scores = model.predict(np.vstack([low, high]))

    assert scores[1] > scores[0]
    assert np.all((scores >= 0.0) & (scores <= 1.0)), "scores stay in [0, 1]"


@requires_sklearn
def test_score_model_json_roundtrip_keeps_predictions(tmp_path: Path) -> None:
    x, y = _training_set()
    model = ScoreModel.fit(x, y, _CONFIG)
    path = tmp_path / "score_model.json"

    model.to_json(path)
    loaded = ScoreModel.from_json(path)

    assert json.loads(path.read_text())["features"] == list(FEATURES), "plain JSON, no pickle"
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    assert loaded.pipeline_config == _CONFIG


@requires_sklearn
def test_score_model_rejects_wrong_feature_count() -> None:
    x, y = _training_set()
    model = ScoreModel.fit(x, y, _CONFIG)

    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, len(FEATURES) - 1)))


@requires_sklearn
def test_score_model_predicts_nothing_for_no_contrail() -> None:
    x, y = _training_set()

    out = ScoreModel.fit(x, y, _CONFIG).predict(np.zeros((0, len(FEATURES))))

    assert out.shape == (0,)


# --- in the pipeline ---


@requires_sklearn
def test_pipeline_with_score_model_replaces_scores_not_masks() -> None:
    probs = (_line(10) | _line(30)).astype(np.float32) * 0.9
    x, y = _training_set()
    model = ScoreModel.fit(x, y, _CONFIG)

    plain_masks, plain_scores = detect_instances_from_probabilities(probs, _CONFIG)
    masks, scores = detect_instances_from_probabilities(probs, _CONFIG, score_model=model)

    assert len(masks) == len(plain_masks) > 0
    assert all(np.array_equal(a, b) for a, b in zip(masks, plain_masks, strict=True))
    expected = model.predict(
        np.stack(
            [
                instance_features(m, probs, s, plain_masks)
                for m, s in zip(plain_masks, plain_scores, strict=True)
            ]
        )
    )
    np.testing.assert_allclose(scores, expected)


@requires_sklearn
def test_pipeline_rejects_a_score_model_trained_with_another_config() -> None:
    x, y = _training_set()
    model = ScoreModel.fit(x, y, _CONFIG)
    other = _CONFIG.model_copy(update={"semantic_threshold": 0.3})

    with pytest.raises(ValueError, match="pipeline config"):
        detect_instances_from_probabilities(
            np.zeros((_S, _S), np.float32), other, score_model=model
        )


@requires_sklearn
def test_score_model_json_records_its_format_version(tmp_path: Path) -> None:
    x, y = _training_set()
    path = tmp_path / "score_model.json"

    ScoreModel.fit(x, y, _CONFIG).to_json(path)

    assert json.loads(path.read_text())["format_version"] == ScoreModel.FORMAT_VERSION


@pytest.mark.parametrize("version", [None, 999])
@requires_sklearn
def test_score_model_rejects_unknown_format_version(tmp_path: Path, version) -> None:
    x, y = _training_set()
    path = tmp_path / "score_model.json"
    ScoreModel.fit(x, y, _CONFIG).to_json(path)
    data = json.loads(path.read_text())
    data.pop("format_version")
    if version is not None:
        data["format_version"] = version
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="format version"):
        ScoreModel.from_json(path)


def test_score_model_fit_without_scikit_learn_names_the_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    x, y = _training_set(10)

    with pytest.raises(ImportError, match=r"deepcontrail-instance\[scores\]"):
        ScoreModel.fit(x, y, _CONFIG)


@requires_sklearn
def test_score_model_records_what_it_was_trained_on(tmp_path: Path) -> None:
    x, y = _training_set()
    path = tmp_path / "score_model.json"

    ScoreModel.fit(x, y, _CONFIG, trained_on={"tta": True, "scene_shape": [64, 64]}).to_json(path)
    trained_on = ScoreModel.from_json(path).trained_on

    assert trained_on["tta"] is True
    assert trained_on["scene_shape"] == [64, 64]
    assert trained_on["scikit_learn_version"], "the fitted trees depend on scikit-learn's version"
    assert trained_on["deepcontrail_instance_version"], "and on the features of this version"


def test_score_model_file_without_a_record_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    ScoreModel(0.5, (), _CONFIG).to_json(path)
    data = json.loads(path.read_text())
    del data["trained_on"]
    path.write_text(json.dumps(data))

    assert ScoreModel.from_json(path).trained_on == {}, "files written before the record load"


def test_pipeline_rejects_a_score_model_trained_on_another_scene_size() -> None:
    model = ScoreModel(0.5, (), _CONFIG, trained_on={"scene_shape": [32, 32]})

    with pytest.raises(ValueError, match="scene"):
        detect_instances_from_probabilities(np.zeros((_S, _S), dtype=np.float32), _CONFIG, model)
