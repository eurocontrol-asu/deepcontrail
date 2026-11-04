"""Pydantic configuration model for Mannstein instance detection.

Serializable via ``model_dump()``. Uses only pydantic + pyyaml — no torch,
scipy, or numpy at module level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

FORMAT_VERSION = 1
"""Version of the instance config file layout (``format_version`` in the YAML files)."""

_SECTIONS = ("format_version", "mannstein", "pipeline")


def _read_sections(path: str | Path) -> dict[str, Any]:
    """Read an instance config YAML and check its top-level keys and format version."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}; expected {list(_SECTIONS)}")
    version = raw.get("format_version", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {version!r} is not supported ({FORMAT_VERSION})")
    return raw


class MannsteinConfig(BaseModel):
    """Settings of the Mannstein step that splits a binary contrail mask into contrails.

    The defaults are the tuned settings of the default pipeline (``InstancePipelineConfig()``),
    chosen for one contrail per annotator-drawn contrail (pooled mAP over IoU 0.25-0.75).

    Parameters
    ----------
    line_filter
        Keep only mask pixels where an oriented line kernel responds above
        ``response_threshold`` before splitting. Off by default: at the tuned settings it keeps
        every pixel of the mask and takes most of the run time.
    num_kernels
        Number of oriented line kernels (line filter only).
    kernel_size
        Kernel size in pixels, odd (line filter only).
    sigma1
        Inner Gaussian sigma in pixels (line filter only).
    sigma_delta
        Outer minus inner Gaussian sigma, in pixels (line filter only).
    response_threshold
        Minimum kernel response (line filter only).
    min_pixels
        Minimum contrail size in pixels.
    min_length
        Minimum contrail length in pixels.
    min_r_score
        Minimum straightness, squared before the comparison: a contrail is kept when its
        straightness ((λ1 - λ2) / (λ1 + λ2))², from the eigenvalues of its pixel-coordinate
        covariance, exceeds ``min_r_score²``. The measure does not depend on the direction.
    lookback
        Number of most recent pixels of a growing contrail used to fit its local direction.
    alignment_threshold
        A neighbouring pixel joins a growing contrail when it lies within this many pixels of
        the contrail's local direction.
    fit_avg_size
        Pixels averaged at each end of a contrail when measuring its length, for the
        length/straightness filter and the minimum size of a scored contrail.
    filter_avg_size
        Overrides ``fit_avg_size`` for the length/straightness filter only. ``None`` reuses
        ``fit_avg_size``.
    iou_threshold
        Two candidates overlapping more than this IoU are duplicates; the first found is kept.
    score_saturation
        Pixel count at which the size part of the score, tanh(n_pixels / score_saturation),
        approaches 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line_filter: bool = False
    num_kernels: Annotated[int, Field(gt=0)] = 24
    kernel_size: Annotated[int, Field(gt=0)] = 25
    sigma1: Annotated[float, Field(gt=0.0)] = 3.38996845929638
    sigma_delta: Annotated[float, Field(gt=0.0)] = 0.7008195042725893
    response_threshold: Annotated[float, Field(ge=0.0)] = 0.00012714987768293402
    min_pixels: Annotated[int, Field(gt=0)] = 5
    min_length: Annotated[float, Field(gt=0.0)] = 4.173422506229322
    min_r_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.07830797096507566
    lookback: Annotated[int, Field(gt=0)] = 53
    alignment_threshold: Annotated[float, Field(gt=0.0)] = 2.4559123612088265
    fit_avg_size: Annotated[int, Field(gt=0)] = 2
    filter_avg_size: Annotated[int, Field(gt=0)] | None = 2
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.29602142259052816
    score_saturation: Annotated[float, Field(gt=0.0)] = 26.486876651437136

    @model_validator(mode="after")
    def _validate_kernel_size(self) -> MannsteinConfig:
        """Ensure kernel_size is odd."""
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MannsteinConfig:
        """Build a ``MannsteinConfig`` from a plain dictionary."""
        return cls.model_validate(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MannsteinConfig:
        """Load a ``MannsteinConfig`` from the ``mannstein:`` section of a YAML file.

        Fields missing from the section keep their defaults; unknown fields are an error.

        Raises
        ------
        ValueError
            If the file has no ``mannstein:`` section, unknown sections or fields, or another
            ``format_version``.
        """
        sections = _read_sections(path)
        if "mannstein" not in sections:
            raise ValueError(f"{path} has no mannstein: section")
        return cls.model_validate({**cls().model_dump(), **sections["mannstein"]})


# Mannstein settings of the consensus preset: tuned with scripts/optimize_mannstein.py on each
# annotator's own mask (train), then chosen on validation predictions for the consensus target
# (see configs/instance_pipeline_consensus.yaml).
_CONSENSUS_MANNSTEIN = MannsteinConfig(
    num_kernels=27,
    kernel_size=15,
    sigma1=1.6281154034079715,
    sigma_delta=0.6895797586480001,
    response_threshold=0.00012114447935536606,
    min_pixels=10,
    min_length=3.6267323822860877,
    min_r_score=0.05470144383145452,
    lookback=53,
    alignment_threshold=2.663708517028666,
    fit_avg_size=3,
    filter_avg_size=2,
    iou_threshold=0.41435493420805575,
    score_saturation=18.325171813375093,
)


class InstancePipelineConfig(BaseModel):
    """Configuration of the probability-map instance pipeline.

    The pipeline thresholds a semantic probability map, splits the binary mask into
    contrails with Mannstein, grows each contrail into neighbouring pixels of moderate
    probability, and scores it. The defaults are the tuned pipeline, and their Mannstein part
    is ``MannsteinConfig()``.

    Parameters
    ----------
    semantic_threshold
        Pixels with probability above this value form the binary mask given to Mannstein.
    grow_threshold
        Growth enters neighbouring pixels with probability above this value.
    grow_steps
        Number of one-pixel growth steps (0 disables growth).
    grow_connectivity
        Neighbourhood used for growth: 4 or 8.
    probability_scores
        Multiply each Mannstein score by the contrail's mean probability.
    mannstein
        Settings of the Mannstein splitting step.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantic_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.46466753022859164
    grow_threshold: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.3711273368123679
    grow_steps: Annotated[int, Field(ge=0)] = 1
    grow_connectivity: Literal[4, 8] = 8
    probability_scores: bool = True
    mannstein: MannsteinConfig = MannsteinConfig()

    @classmethod
    def consensus(cls) -> InstancePipelineConfig:
        """Return the settings for contrails agreed by most annotators.

        Same as ``configs/instance_pipeline_consensus.yaml``.

        No growth: growth widens contrails toward how single annotators draw them, which helps
        when each annotator is the ground truth (the defaults) but not for the consensus.
        """
        return cls(
            semantic_threshold=0.5,
            grow_threshold=0.4,
            grow_steps=0,
            grow_connectivity=8,
            probability_scores=True,
            mannstein=_CONSENSUS_MANNSTEIN,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstancePipelineConfig:
        """Build an ``InstancePipelineConfig`` from a plain dictionary."""
        return cls.model_validate(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> InstancePipelineConfig:
        """Load from a YAML file with ``pipeline:`` and ``mannstein:`` sections.

        Either section may be missing or partial: what is missing is taken from the preset
        ``InstancePipelineConfig()``, so ``mannstein: {num_kernels: 24}`` changes only
        ``num_kernels``. Unknown sections or fields are an error, so a typo is never silently
        replaced by a default.

        Raises
        ------
        ValueError
            On unknown sections or fields, invalid values, or another ``format_version``.
        """
        sections = _read_sections(path)
        preset = cls().model_dump()
        mannstein = {**preset.pop("mannstein"), **sections.get("mannstein", {})}
        return cls.model_validate(
            {**preset, **sections.get("pipeline", {}), "mannstein": mannstein}
        )

    def to_yaml(self, path: str | Path) -> None:
        """Write ``format_version`` and the ``mannstein:`` and ``pipeline:`` sections."""
        data = self.model_dump()
        mannstein = data.pop("mannstein")
        with Path(path).open("w") as f:
            yaml.safe_dump(
                {"format_version": FORMAT_VERSION, "mannstein": mannstein, "pipeline": data},
                f,
                sort_keys=False,
            )
