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
    match_quality,
    scene_features,
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


def _tree() -> dict[str, np.ndarray]:
    """Build a minimal valid tree: the root (node 0) branches to two leaves (nodes 1 and 2)."""
    return {
        "feature_idx": np.array([0, 0, 0]),
        "num_threshold": np.array([0.5, 0.0, 0.0]),
        "missing_go_to_left": np.array([True, False, False]),
        "left": np.array([1, 0, 0]),
        "right": np.array([2, 0, 0]),
        "is_leaf": np.array([False, True, True]),
        "value": np.array([0.0, 0.1, -0.1]),
    }


def _tree_model() -> ScoreModel:
    """Build a one-tree score model without needing scikit-learn, to exercise from_json."""
    return ScoreModel(0.5, (_tree(),), _CONFIG)


# --- features and labels ---


def test_scene_features_describe_size_shape_and_probability() -> None:
    mask = _line(10, width=2)
    probs = np.where(mask, 0.8, 0.1).astype(np.float32)

    feats = scene_features([mask], probs, [0.6])[0]

    named = dict(zip(FEATURES, feats, strict=True))
    assert named["n_pixels"] == 96
    assert named["length"] == pytest.approx(48.0, rel=0.05)
    assert named["width"] == pytest.approx(2.0, rel=0.05)
    assert named["mean_prob"] == pytest.approx(0.8)
    assert named["ring_mean_prob"] == pytest.approx(0.1), "pixels just around the contrail"
    assert named["contrast"] == pytest.approx(0.7), "mean probability minus the ring's"
    assert named["pipeline_score"] == pytest.approx(0.6)
    assert named["n_scene_instances"] == 1


@pytest.mark.parametrize(
    ("pixels", "sin2", "cos2"),
    [
        ((np.full(48, 10), np.arange(8, 56)), 0.0, 1.0),  # horizontal
        ((np.arange(8, 56), np.full(48, 30)), 0.0, -1.0),  # vertical
        ((np.arange(8, 56), np.arange(8, 56)), 1.0, 0.0),  # diagonal
    ],
)
def test_scene_features_of_a_straight_line_give_its_orientation(pixels, sin2, cos2) -> None:
    mask = np.zeros((_S, _S), dtype=bool)
    mask[pixels] = True
    probs = mask.astype(np.float32)

    named = dict(zip(FEATURES, scene_features([mask], probs, [0.5])[0], strict=True))

    assert named["straightness"] == pytest.approx(1.0), "a one-pixel line is perfectly straight"
    assert named["orientation_sin2"] == pytest.approx(sin2, abs=1e-12)
    assert named["orientation_cos2"] == pytest.approx(cos2, abs=1e-12)


def test_scene_features_of_a_square_blob_have_no_straightness() -> None:
    mask = np.zeros((_S, _S), dtype=bool)
    mask[20:25, 20:25] = True

    named = dict(
        zip(FEATURES, scene_features([mask], mask.astype(np.float32), [0.5])[0], strict=True)
    )

    assert named["straightness"] == pytest.approx(0.0, abs=1e-12)


def test_scene_features_summarise_the_probabilities_inside_the_contrail() -> None:
    mask = _line(10, start=0, stop=10)  # 10 pixels, centre (10, 4.5)
    values = np.arange(0.05, 1.0, 0.1)  # 0.05, 0.15, ..., 0.95
    probs = np.zeros((_S, _S), dtype=np.float32)
    probs[10, 0:10] = values

    named = dict(zip(FEATURES, scene_features([mask], probs, [0.5])[0], strict=True))

    assert named["mean_prob"] == pytest.approx(0.5, abs=1e-6)
    assert named["min_prob"] == pytest.approx(0.05, abs=1e-6)
    assert named["max_prob"] == pytest.approx(0.95, abs=1e-6)
    assert named["std_prob"] == pytest.approx(np.std(values), abs=1e-6)
    assert named["p10_prob"] == pytest.approx(0.14, abs=1e-6)
    assert named["p90_prob"] == pytest.approx(0.86, abs=1e-6)
    assert named["frac_above_0_7"] == pytest.approx(0.3), "0.75, 0.85 and 0.95"
    assert named["ring_mean_prob"] == 0.0
    assert named["contrast"] == pytest.approx(0.5, abs=1e-6)
    assert named["border_distance"] == pytest.approx(4.5 / _S), "nearest edge of the centre"


def test_scene_features_describe_the_scene_the_same_for_every_contrail() -> None:
    masks = [_line(10), _line(40, width=3)]  # 48 + 144 pixels, disjoint
    probs = np.where(masks[0] | masks[1], 0.8, 0.0).astype(np.float32)

    features = scene_features(masks, probs, [0.5, 0.5])

    named = [dict(zip(FEATURES, row, strict=True)) for row in features]
    assert [n["n_scene_instances"] for n in named] == [2, 2]
    assert [n["scene_pred_area"] for n in named] == [192, 192]
    assert [n["shared_pixels_frac"] for n in named] == [0.0, 0.0]


def test_scene_features_treat_integer_masks_like_boolean_masks() -> None:
    mask_u = _line(10).astype(np.uint8)
    other_u = _line(30).astype(np.uint8)
    probs = np.where(mask_u | other_u, 0.8, 0.0).astype(np.float32)

    feats_u = scene_features([mask_u, other_u], probs, [0.5, 0.5])[0]
    mask_b, other_b = mask_u.astype(bool), other_u.astype(bool)
    feats_b = scene_features([mask_b, other_b], probs, [0.5, 0.5])[0]

    np.testing.assert_allclose(
        feats_u, feats_b, err_msg="a 0/1 int mask must select the same pixels as a bool mask"
    )
    assert dict(zip(FEATURES, feats_u, strict=True))["mean_prob"] == pytest.approx(0.8)


def test_scene_features_count_shared_pixels_with_other_contrails_only() -> None:
    row = _line(10)  # 48 pixels on row 10
    column = np.zeros((_S, _S), dtype=bool)
    column[:, 30] = True  # crosses the row at one pixel
    probs = np.where(row | column, 0.8, 0.1).astype(np.float32)

    alone = scene_features([row], probs, [0.5])
    crossing = scene_features([row, column], probs, [0.5, 0.5])
    twice = scene_features([row, row.copy()], probs, [0.5, 0.5])

    shared = FEATURES.index("shared_pixels_frac")
    assert alone[0, shared] == 0.0, "a contrail never overlaps itself"
    assert crossing[0, shared] == pytest.approx(1 / 48)
    assert twice[0, shared] == 1.0, "two detections covering the same pixels share them all"


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
    expected = model.predict(scene_features(plain_masks, probs, plain_scores))
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


@requires_sklearn
def test_score_model_from_newer_library_asks_to_upgrade(tmp_path: Path) -> None:
    x, y = _training_set()
    path = tmp_path / "score_model.json"
    ScoreModel.fit(x, y, _CONFIG).to_json(path)
    data = json.loads(path.read_text())
    data["format_version"] = 3
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="upgrade deepcontrail-instance"):
        ScoreModel.from_json(path)


def test_score_model_reads_a_format_1_file(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    data["format_version"] = 1
    data["pipeline_config"]["mannstein"]["min_r_score"] = 0.5
    del data["pipeline_config"]["mannstein"]["min_straightness"]
    path.write_text(json.dumps(data))

    loaded = ScoreModel.from_json(path)

    assert loaded.pipeline_config.mannstein.min_straightness == 0.25, "format 1 compared r squared"


@pytest.mark.parametrize("version", [None, 0, "1"])
@requires_sklearn
def test_score_model_without_format_version_is_refused(tmp_path: Path, version) -> None:
    x, y = _training_set()
    path = tmp_path / "score_model.json"
    ScoreModel.fit(x, y, _CONFIG).to_json(path)
    data = json.loads(path.read_text())
    data.pop("format_version")
    if version is not None:
        data["format_version"] = version
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="not supported"):
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


def test_score_model_from_json_names_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    del data["baseline"]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="baseline"):
        ScoreModel.from_json(path)


def test_score_model_from_json_rejects_other_feature_names(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    data["features"] = data["features"][::-1]
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="features differ"):
        ScoreModel.from_json(path)


def test_score_model_from_json_rejects_a_non_finite_threshold(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    data["trees"][0]["num_threshold"][0] = float("nan")  # json writes NaN
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="not finite"):
        ScoreModel.from_json(path)


# --- from_json rejects a malformed tree (a downloaded model may be corrupt) ---


def test_score_model_from_json_rejects_a_tree_whose_child_points_to_itself(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    data["trees"][0]["left"][0] = (
        0  # the root's left child is the root: predict() would loop forever
    )
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="tree 0"):
        ScoreModel.from_json(path)


def test_score_model_from_json_rejects_a_negative_feature_index(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    data["trees"][0]["feature_idx"][0] = -1  # a negative index would read the last feature silently
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="tree 0"):
        ScoreModel.from_json(path)


def test_score_model_from_json_rejects_tree_arrays_of_different_lengths(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    data["trees"][0]["value"] = data["trees"][0]["value"][:-1]  # ragged: would fail mid-prediction
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="tree 0"):
        ScoreModel.from_json(path)


def test_score_model_from_json_rejects_a_non_finite_leaf_value(tmp_path: Path) -> None:
    path = tmp_path / "score_model.json"
    _tree_model().to_json(path)
    data = json.loads(path.read_text())
    data["trees"][0]["value"][1] = float("nan")
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="tree 0"):
        ScoreModel.from_json(path)


def test_score_model_from_json_accepts_a_model_written_by_to_json(tmp_path: Path) -> None:
    model = _tree_model()
    path = tmp_path / "score_model.json"
    x = np.random.default_rng(0).random((5, len(FEATURES)))

    model.to_json(path)
    loaded = ScoreModel.from_json(path)

    np.testing.assert_array_equal(
        loaded.predict(x), model.predict(x), "a well-formed model loads and predicts unchanged"
    )


def test_pipeline_rejects_a_score_model_trained_on_another_scene_size() -> None:
    model = ScoreModel(0.5, (), _CONFIG, trained_on={"scene_shape": [32, 32]})

    with pytest.raises(ValueError, match="scene"):
        detect_instances_from_probabilities(np.zeros((_S, _S), dtype=np.float32), _CONFIG, model)
