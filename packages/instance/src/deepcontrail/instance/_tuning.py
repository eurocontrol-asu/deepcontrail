"""Search spaces of the instance tuning scripts (Optuna trials).

Private module (leading underscore): it serves the repository scripts in ``scripts/`` and is
not part of the public API, so it may change between releases.

Each ``suggest_*`` function builds a config from a trial; each ``*_trial_params`` function turns
a config back into the trial parameters, so known configs can be enqueued as starting trials.
Optuna is only needed by the caller: nothing here imports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from deepcontrail.instance.config import InstancePipelineConfig, MannsteinConfig

if TYPE_CHECKING:
    from optuna.trial import BaseTrial


def suggest_mannstein_config(trial: BaseTrial) -> MannsteinConfig:
    """Build a ``MannsteinConfig`` from a trial; the line filter stays off.

    The line-filter settings are not searched: the filter is off in every search, as it kept
    every mask pixel at the tuned settings. ``filter_avg_size`` is searched separately from
    ``fit_avg_size``, so the length filter and the score can average a different number of end
    pixels. Parameters are suggested in a fixed order, which determines what a seeded sampler
    draws.
    """
    return MannsteinConfig(
        line_filter=False,
        min_pixels=trial.suggest_int("min_pixels", 3, 50),
        min_length=trial.suggest_float("min_length", 3.0, 50.0),
        min_straightness=trial.suggest_float("min_straightness", 0.0025, 0.36),
        lookback=trial.suggest_int("lookback", 10, 60),
        alignment_threshold=trial.suggest_float("alignment_threshold", 0.5, 5.0),
        fit_avg_size=trial.suggest_int("fit_avg_size", 1, 10),
        iou_threshold=trial.suggest_float("iou_threshold", 0.2, 0.8),
        score_saturation=trial.suggest_float("score_saturation", 10.0, 100.0),
        filter_avg_size=trial.suggest_int("filter_avg_size", 1, 10),
    )


_SEARCHED = (
    "min_pixels",
    "min_length",
    "min_straightness",
    "lookback",
    "alignment_threshold",
    "fit_avg_size",
    "iou_threshold",
    "score_saturation",
    "filter_avg_size",
)


def mannstein_trial_params(config: MannsteinConfig) -> dict[str, Any]:
    """Return the searched settings of a config as trial parameters.

    ``filter_avg_size=None`` becomes ``fit_avg_size``; the line-filter settings are dropped.
    """
    params = {name: getattr(config, name) for name in _SEARCHED}
    if params["filter_avg_size"] is None:
        params["filter_avg_size"] = params["fit_avg_size"]
    return params


def suggest_pipeline_config(trial: BaseTrial) -> InstancePipelineConfig:
    """Build an ``InstancePipelineConfig`` from a trial; scores are always probability-weighted.

    The Mannstein part is :func:`suggest_mannstein_config`, suggested first.
    """
    mannstein = suggest_mannstein_config(trial)
    return InstancePipelineConfig(
        semantic_threshold=trial.suggest_float("semantic_threshold", 0.3, 0.7),
        grow_threshold=trial.suggest_float("grow_threshold", 0.2, 0.6),
        grow_steps=trial.suggest_int("grow_steps", 0, 3),
        grow_connectivity=cast(
            "Literal[4, 8]", trial.suggest_categorical("grow_connectivity", [4, 8])
        ),
        probability_scores=True,
        mannstein=mannstein,
    )


def pipeline_trial_params(config: InstancePipelineConfig) -> dict[str, Any]:
    """Flatten a pipeline config into the trial parameter names of the pipeline search."""
    params = mannstein_trial_params(config.mannstein)
    params.update(
        {
            "semantic_threshold": config.semantic_threshold,
            "grow_threshold": config.grow_threshold,
            "grow_steps": config.grow_steps,
            "grow_connectivity": config.grow_connectivity,
        }
    )
    return params
