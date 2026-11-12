"""The instance step stored in a pretrained semantic bundle.

A pretrained bundle (``deepcontrail.segmentation.load_pretrained``) may carry the instance step
tuned for its model, as two optional files next to the weights:

- ``instance_pipeline.yaml``: the :class:`InstancePipelineConfig` to run on its probabilities;
- ``score_model.json``: a :class:`ScoreModel` trained on that model's predictions.

This module reads and writes them from a folder only, so ``deepcontrail.instance`` never imports
``deepcontrail.segmentation``::

    model = load_pretrained("eurocontrol-asu/deepcontrail-segmentation")
    step = load_instance_step(model.directory, tta=True)
    masks, scores = step(predict(model, composite, tta=True))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepcontrail.instance.config import InstancePipelineConfig
from deepcontrail.instance.pipeline import detect_instances_from_probabilities
from deepcontrail.instance.scoring import ScoreModel

if TYPE_CHECKING:
    import numpy as np

PIPELINE_FILE = "instance_pipeline.yaml"
SCORE_MODEL_FILE = "score_model.json"


@dataclass(frozen=True)
class InstanceStep:
    """Pipeline settings and optional learned scores; call it on a probability map."""

    config: InstancePipelineConfig
    score_model: ScoreModel | None = None

    def __call__(self, probs: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        """Run :func:`detect_instances_from_probabilities` with this step's settings."""
        return detect_instances_from_probabilities(probs, self.config, self.score_model)


def load_instance_step(directory: str | Path, *, tta: bool | None = None) -> InstanceStep:
    """Read the instance step of a bundle folder; defaults where its files are absent.

    Parameters
    ----------
    directory
        The bundle folder, e.g. ``Pretrained.directory``.
    tta
        Whether the probabilities will be made with test-time augmentation. A score model that
        records the other setting is refused: TTA changes the probabilities its features are
        computed from. ``None`` skips the check.

    Returns
    -------
    InstanceStep
        ``instance_pipeline.yaml`` (else ``InstancePipelineConfig()``) and ``score_model.json``
        (else ``None``).

    Raises
    ------
    FileNotFoundError
        If ``directory`` does not exist (a typo would otherwise silently drop the learned scores).
    ValueError
        If the score model was trained with other pipeline settings than the bundle's, or with
        the other TTA setting.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory}: no such bundle folder")
    pipeline_path = directory / PIPELINE_FILE
    config = (
        InstancePipelineConfig.from_yaml(pipeline_path)
        if pipeline_path.is_file()
        else InstancePipelineConfig()
    )
    score_path = directory / SCORE_MODEL_FILE
    if not score_path.is_file():
        return InstanceStep(config)
    score_model = ScoreModel.from_json(score_path)
    if score_model.pipeline_config != config:
        raise ValueError(
            f"{score_path} was trained with another pipeline config than {pipeline_path.name}"
        )
    trained_tta = score_model.trained_on.get("tta")
    if tta is not None and trained_tta is not None and trained_tta != tta:
        raise ValueError(
            f"{score_path} was trained on probabilities made with TTA={trained_tta}, "
            f"but they will be made with TTA={tta}"
        )
    return InstanceStep(config, score_model)


def save_instance_step(
    directory: str | Path,
    config: InstancePipelineConfig,
    score_model: ScoreModel | None = None,
) -> None:
    """Write ``instance_pipeline.yaml`` (and ``score_model.json``) into a bundle folder.

    Without ``score_model``, an earlier ``score_model.json`` in the folder is removed, so the
    bundle uses the pipeline's own scores as asked.

    Raises
    ------
    ValueError
        If ``score_model`` was trained with other pipeline settings than ``config``.
    """
    if score_model is not None and score_model.pipeline_config != config:
        raise ValueError("the score model was trained with another pipeline config")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config.to_yaml(directory / PIPELINE_FILE)
    if score_model is not None:
        score_model.to_json(directory / SCORE_MODEL_FILE)
    else:
        (directory / SCORE_MODEL_FILE).unlink(missing_ok=True)
