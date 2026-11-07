"""Pydantic configuration of how scenes are read from a dataset root.

Serializable via ``model_dump()`` → YAML/JSON for experiment logging.
Uses only pydantic + pyyaml so that ``import deepcontrail.core`` stays torch-free.
Task packages extend ``DatasetConfig`` with their own loader fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepcontrail.core.false_color import FalseColorConfig  # noqa: TC001 (pydantic field)


class DatasetConfig(BaseModel):
    """Which scenes to read from a dataset root, and how to build their images.

    Parameters
    ----------
    root
        Path to a dataset root in the DeepContrail format (``spectral_bands/``
        with its ``bands.json``, label directories, ``metadata/``).
    split
        ``"train"``, ``"val"``, or ``"test"``.  Both ``"train"`` and ``"val"``
        read from the ``train/`` directory on disk; the dataset partitions stems
        deterministically using ``val_fraction``.  ``"test"`` reads from ``test/``.
    false_color
        Bands and bounds of the composite built from ``spectral_bands/``, by the
        band keys of the dataset's ``bands.json``. Required: there is no default.
    pixel_shift
        Sub-pixel translation (right and down) applied to images at load time to
        align them with the masks. Use the dataset's ``recommended_pixel_shift``
        (``metadata/dataset_info.json``); default 0.0 (no shift).
    seed
        RNG seed for the train/val partition and for reproducible augmentation.
    val_fraction
        Fraction of training stems held out for validation (seeded by ``seed``).
        Must be > 0 when ``fold`` is not set.  Ignored when ``fold`` is set
        (fold partitioning takes over).
    fold
        K-fold cross-validation fold index, or ``None`` to disable CV.
    num_folds
        Total number of CV folds. Only used when ``fold`` is set.
    """

    model_config = ConfigDict(frozen=True)

    root: str
    split: Literal["train", "val", "test"] = "train"
    false_color: FalseColorConfig
    pixel_shift: Annotated[float, Field(ge=0.0)] = 0.0
    seed: Annotated[int, Field(ge=0)] = 42
    val_fraction: Annotated[float, Field(default=0.1, ge=0.0, lt=1.0)] = 0.1
    fold: int | None = None
    num_folds: Annotated[int, Field(default=4, ge=2)] = 4

    @model_validator(mode="after")
    def _validate_split_strategy(self) -> Self:
        """Validate the split configuration.

        - ``fold`` must be in ``[0, num_folds)`` when set.
        - ``val_fraction`` must be > 0 when ``fold`` is not set.
        """
        if self.fold is not None and not (0 <= self.fold < self.num_folds):
            msg = f"fold {self.fold} out of range [0, {self.num_folds})"
            raise ValueError(msg)
        if self.val_fraction <= 0 and self.fold is None:
            msg = "val_fraction must be > 0 (use fold for CV instead)"
            raise ValueError(msg)
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Build a config from a plain dictionary.

        Pydantic handles nested dict → sub-model parsing automatically.
        """
        return cls.model_validate(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a config from the ``data:`` section of a YAML file."""
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw["data"])
