"""Tests for deepcontrail.instance._tuning (search spaces of the tuning scripts)."""

from __future__ import annotations

import optuna

from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig


def test_suggest_mannstein_config_search_space_is_pinned() -> None:
    from deepcontrail.instance._tuning import suggest_mannstein_config

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    config = suggest_mannstein_config(trial)
    dists = trial.distributions

    assert list(dists) == [
        "min_pixels",
        "min_length",
        "min_r_score",
        "lookback",
        "alignment_threshold",
        "fit_avg_size",
        "iou_threshold",
        "score_saturation",
        "filter_avg_size",
    ], "the order fixes what a seeded sampler draws"
    assert (dists["filter_avg_size"].low, dists["filter_avg_size"].high) == (1, 10)
    assert config.line_filter is False, "the searches run without the line filter"
    assert config.filter_avg_size is not None


def test_mannstein_trial_params_roundtrip_keeps_behaviour() -> None:
    from deepcontrail.instance._tuning import mannstein_trial_params, suggest_mannstein_config

    explicit = MannsteinConfig(fit_avg_size=5, filter_avg_size=3, min_pixels=12)
    implicit = MannsteinConfig(fit_avg_size=4, filter_avg_size=None)

    rebuilt = suggest_mannstein_config(optuna.trial.FixedTrial(mannstein_trial_params(explicit)))
    rebuilt_implicit = suggest_mannstein_config(
        optuna.trial.FixedTrial(mannstein_trial_params(implicit))
    )

    assert rebuilt == explicit
    assert rebuilt_implicit == implicit.model_copy(update={"filter_avg_size": 4}), (
        "filter_avg_size=None means 'same as fit_avg_size'"
    )


def test_mannstein_trial_params_holds_only_searched_settings() -> None:
    from deepcontrail.instance._tuning import mannstein_trial_params

    params = mannstein_trial_params(MannsteinConfig(line_filter=True))

    assert "line_filter" not in params
    assert "num_kernels" not in params


def test_suggest_pipeline_config_bounds_give_valid_configs() -> None:
    from deepcontrail.instance._tuning import suggest_pipeline_config

    for bound in ("low", "high"):
        params = MannsteinConfig().model_dump()
        params.update(
            {
                "semantic_threshold": 0.3 if bound == "low" else 0.7,
                "grow_threshold": 0.2 if bound == "low" else 0.6,
                "grow_steps": 0 if bound == "low" else 3,
                "grow_connectivity": 4 if bound == "low" else 8,
                "filter_avg_size": 1 if bound == "low" else 10,
            }
        )

        config = suggest_pipeline_config(optuna.trial.FixedTrial(params))

        assert isinstance(config, InstancePipelineConfig)
        assert config.semantic_threshold == params["semantic_threshold"]
        assert config.grow_steps == params["grow_steps"]
        assert config.mannstein.filter_avg_size == params["filter_avg_size"]
        assert config.probability_scores, "scores are always probability-weighted"


def test_pipeline_search_space_ranges_are_pinned() -> None:
    from deepcontrail.instance._tuning import suggest_pipeline_config

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()
    suggest_pipeline_config(trial)
    dists = trial.distributions

    assert (dists["semantic_threshold"].low, dists["semantic_threshold"].high) == (0.3, 0.7)
    assert (dists["grow_threshold"].low, dists["grow_threshold"].high) == (0.2, 0.6)
    assert (dists["grow_steps"].low, dists["grow_steps"].high) == (0, 3)
    assert (dists["filter_avg_size"].low, dists["filter_avg_size"].high) == (1, 10)
    assert list(dists)[-4:] == [
        "semantic_threshold",
        "grow_threshold",
        "grow_steps",
        "grow_connectivity",
    ], "Mannstein settings are suggested first"


def test_pipeline_trial_params_roundtrip_through_suggest() -> None:
    from deepcontrail.instance._tuning import pipeline_trial_params, suggest_pipeline_config

    config = InstancePipelineConfig(
        semantic_threshold=0.45,
        grow_threshold=0.35,
        grow_steps=2,
        grow_connectivity=8,
        mannstein=MannsteinConfig(min_pixels=9, fit_avg_size=4, filter_avg_size=3),
    )
    implicit = config.model_copy(
        update={"mannstein": config.mannstein.model_copy(update={"filter_avg_size": None})}
    )

    rebuilt = suggest_pipeline_config(optuna.trial.FixedTrial(pipeline_trial_params(config)))
    rebuilt_implicit = suggest_pipeline_config(
        optuna.trial.FixedTrial(pipeline_trial_params(implicit))
    )

    assert rebuilt == config
    assert rebuilt_implicit.mannstein.filter_avg_size == implicit.mannstein.fit_avg_size, (
        "filter_avg_size=None means 'same as fit_avg_size'"
    )
