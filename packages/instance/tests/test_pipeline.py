"""Tests for the probability-map instance pipeline (config + detect_instances_from_probabilities)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from deepcontrail.instance import (
    InstancePipelineConfig,
    MannsteinConfig,
    detect_instances,
    detect_instances_from_probabilities,
)

if TYPE_CHECKING:
    from pathlib import Path

H = W = 64


def _line_probs(core: float = 0.9, flank: float = 0.45) -> np.ndarray:
    """Build a horizontal contrail: core row at ``core``, the rows above and below at ``flank``."""
    probs = np.zeros((H, W), dtype=np.float32)
    probs[31, 5:56] = flank
    probs[33, 5:56] = flank
    probs[32, 5:56] = core
    return probs


# --- InstancePipelineConfig ---


def test_pipeline_config_defaults_are_valid_and_frozen():
    cfg = InstancePipelineConfig()

    assert 0.0 < cfg.semantic_threshold < 1.0
    assert isinstance(cfg.mannstein, MannsteinConfig)
    with pytest.raises(ValidationError):
        cfg.grow_steps = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_value",
    [
        {"semantic_threshold": 0.0},
        {"semantic_threshold": 1.0},
        {"grow_threshold": -0.1},
        {"grow_steps": -1},
        {"grow_connectivity": 6},
    ],
)
def test_pipeline_config_rejects_invalid_values(field_value):
    with pytest.raises(ValidationError):
        InstancePipelineConfig(**field_value)


def test_pipeline_config_from_yaml_reads_pipeline_and_mannstein_sections(tmp_path: Path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        yaml.dump(
            {
                "mannstein": {"num_kernels": 30, "kernel_size": 9, "line_filter": True},
                "pipeline": {"semantic_threshold": 0.45, "grow_threshold": 0.35, "grow_steps": 2},
            }
        )
    )

    cfg = InstancePipelineConfig.from_yaml(path)

    assert cfg.semantic_threshold == 0.45
    assert cfg.grow_threshold == 0.35
    assert cfg.grow_steps == 2
    assert cfg.mannstein.num_kernels == 30
    assert cfg.mannstein.kernel_size == 9


def test_pipeline_config_from_yaml_without_pipeline_section_uses_defaults(tmp_path: Path):
    path = tmp_path / "mannstein_only.yaml"
    path.write_text(yaml.dump({"mannstein": {"num_kernels": 20}}))

    cfg = InstancePipelineConfig.from_yaml(path)

    assert cfg.mannstein.num_kernels == 20
    assert cfg.grow_steps == InstancePipelineConfig().grow_steps


def test_pipeline_config_to_yaml_roundtrips(tmp_path: Path):
    cfg = InstancePipelineConfig(semantic_threshold=0.42, grow_steps=2)
    path = tmp_path / "out.yaml"

    cfg.to_yaml(path)

    assert InstancePipelineConfig.from_yaml(path) == cfg
    assert MannsteinConfig.from_yaml(path) == cfg.mannstein, (
        "the mannstein section stays loadable alone"
    )


# --- detect_instances_from_probabilities ---


def test_detect_from_probabilities_empty_map_returns_empty_lists():
    masks, scores = detect_instances_from_probabilities(np.zeros((H, W), np.float32))

    assert masks == []
    assert scores == []


def test_detect_from_probabilities_without_refinement_matches_mannstein():
    probs = _line_probs()
    cfg = InstancePipelineConfig(semantic_threshold=0.5, grow_steps=0, probability_scores=False)

    masks, scores = detect_instances_from_probabilities(probs, cfg)
    ref_masks, ref_scores = detect_instances((probs > 0.5).astype(np.uint8), cfg.mannstein)

    assert len(masks) == len(ref_masks)
    assert all(np.array_equal(a, b) for a, b in zip(masks, ref_masks, strict=True))
    assert scores == pytest.approx(ref_scores)


def test_detect_from_probabilities_grows_into_flanks_above_grow_threshold():
    probs = _line_probs(core=0.9, flank=0.45)
    grow = InstancePipelineConfig(semantic_threshold=0.5, grow_threshold=0.4, grow_steps=1)
    no_grow = InstancePipelineConfig(semantic_threshold=0.5, grow_threshold=0.5, grow_steps=1)

    grown, _ = detect_instances_from_probabilities(probs, grow)
    plain, _ = detect_instances_from_probabilities(probs, no_grow)

    assert len(grown) == 1
    assert len(plain) == 1
    assert grown[0][31, 20], "flank pixels above grow_threshold are added"
    assert grown[0][33, 20], "flank pixels above grow_threshold are added"
    assert not plain[0][31, 20], "flank pixels at or below grow_threshold are not added"


def test_detect_from_probabilities_scores_are_probability_weighted():
    probs = _line_probs(core=0.6, flank=0.0)
    weighted = InstancePipelineConfig(grow_steps=0, probability_scores=True)
    plain = InstancePipelineConfig(grow_steps=0, probability_scores=False)

    _, w_scores = detect_instances_from_probabilities(probs, weighted)
    _, p_scores = detect_instances_from_probabilities(probs, plain)

    assert w_scores == pytest.approx([s * 0.6 for s in p_scores])
    assert all(0.0 <= s <= 1.0 for s in w_scores)


def test_detect_from_probabilities_masks_are_boolean_with_input_shape():
    masks, _ = detect_instances_from_probabilities(_line_probs())

    assert masks
    assert all(m.dtype == bool for m in masks)
    assert all(m.shape == (H, W) for m in masks)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, H, W), np.float32),
        np.full((H, W), 1.5, np.float32),
        np.full((H, W), -0.2, np.float32),
        np.where(np.eye(H, W, dtype=bool), np.nan, 0.5).astype(np.float32),
    ],
)
def test_detect_from_probabilities_rejects_invalid_maps(bad):
    with pytest.raises(ValueError, match="probabilit"):
        detect_instances_from_probabilities(bad)


def test_pipeline_config_defaults_match_shipped_config():
    from pathlib import Path

    shipped = Path(__file__).resolve().parents[1] / "configs" / "instance_pipeline.yaml"

    assert InstancePipelineConfig.from_yaml(shipped) == InstancePipelineConfig(), (
        "configs/instance_pipeline.yaml and the InstancePipelineConfig defaults must stay identical"
    )


def test_pipeline_config_defaults_are_the_tuned_pipeline():
    cfg = InstancePipelineConfig()

    assert cfg.semantic_threshold == pytest.approx(0.4647, abs=1e-4), "tuned cut, below 0.5"
    assert cfg.grow_steps == 1
    assert cfg.grow_threshold == pytest.approx(0.3711, abs=1e-4)
    assert cfg.grow_connectivity == 8
    assert cfg.probability_scores
    assert cfg.mannstein == MannsteinConfig(), "the Mannstein defaults are the pipeline's"
    assert cfg.mannstein.line_filter is False, "the line filter keeps every pixel: skip it"


def test_shipped_consensus_config_loads_and_does_not_grow():
    from pathlib import Path

    shipped = Path(__file__).resolve().parents[1] / "configs" / "instance_pipeline_consensus.yaml"

    cfg = InstancePipelineConfig.from_yaml(shipped)

    assert cfg.grow_steps == 0, (
        "growth widens contrails toward single annotators, not the consensus"
    )
    assert cfg.probability_scores
    assert cfg != InstancePipelineConfig()


def test_consensus_preset_matches_shipped_config_without_a_repository_file():
    from pathlib import Path

    shipped = Path(__file__).resolve().parents[1] / "configs" / "instance_pipeline_consensus.yaml"

    preset = InstancePipelineConfig.consensus()

    assert preset == InstancePipelineConfig.from_yaml(shipped), (
        "the preset is what an installed package offers; the YAML documents it"
    )
    assert preset.grow_steps == 0
    assert preset != InstancePipelineConfig()


# --- strict config files ---


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(data))
    return path


@pytest.mark.parametrize(
    "data",
    [
        {"pipeline": {"grow_treshold": 0.9}},
        {"mannstein": {"num_kernel": 24}},
        {"pipline": {"grow_steps": 0}},
    ],
)
def test_pipeline_config_yaml_rejects_unknown_keys(tmp_path: Path, data: dict):
    with pytest.raises(ValueError, match=r"grow_treshold|num_kernel|pipline"):
        InstancePipelineConfig.from_yaml(_write_yaml(tmp_path, data))


def test_pipeline_config_partial_mannstein_section_completes_from_the_preset(tmp_path: Path):
    cfg = InstancePipelineConfig.from_yaml(_write_yaml(tmp_path, {"mannstein": {"num_kernels": 9}}))

    preset = InstancePipelineConfig().mannstein
    assert cfg.mannstein.num_kernels == 9
    assert cfg.mannstein.kernel_size == preset.kernel_size, "not some other default"
    assert cfg.mannstein.model_dump(exclude={"num_kernels"}) == preset.model_dump(
        exclude={"num_kernels"}
    )


def test_pipeline_config_yaml_without_mannstein_section_uses_the_preset(tmp_path: Path):
    cfg = InstancePipelineConfig.from_yaml(_write_yaml(tmp_path, {"pipeline": {"grow_steps": 0}}))

    assert cfg.grow_steps == 0
    assert cfg.mannstein == InstancePipelineConfig().mannstein


def test_pipeline_config_yaml_empty_mannstein_section_uses_the_preset(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("format_version: 2\nmannstein:\npipeline:\n  grow_steps: 0\n")

    cfg = InstancePipelineConfig.from_yaml(path)

    assert cfg == InstancePipelineConfig(grow_steps=0), (
        "an empty 'mannstein:' section (None) falls back to the preset, not an error"
    )


def test_pipeline_config_yaml_must_be_a_mapping(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("- a\n")

    with pytest.raises(ValueError, match="mapping"):
        InstancePipelineConfig.from_yaml(path)


def test_pipeline_config_yaml_records_and_checks_its_format_version(tmp_path: Path):
    path = tmp_path / "out.yaml"
    InstancePipelineConfig().to_yaml(path)

    assert yaml.safe_load(path.read_text())["format_version"] == 2
    with pytest.raises(ValueError, match="upgrade deepcontrail-instance"):
        InstancePipelineConfig.from_yaml(_write_yaml(tmp_path, {"format_version": 3}))
    with pytest.raises(ValueError, match="not supported"):
        InstancePipelineConfig.from_yaml(_write_yaml(tmp_path, {"format_version": 0}))


def test_pipeline_config_reads_a_format_1_file_with_min_r_score(tmp_path: Path):
    path = tmp_path / "v1.yaml"
    path.write_text("format_version: 1\nmannstein:\n  min_r_score: 0.3\n")

    config = InstancePipelineConfig.from_yaml(path)

    assert config.mannstein.min_straightness == 0.3**2, "format 1 compared r squared"


def test_pipeline_config_reads_an_unversioned_file_with_min_r_score_as_format_1(tmp_path: Path):
    path = tmp_path / "old.yaml"
    path.write_text("mannstein:\n  min_r_score: 0.3\n")

    config = InstancePipelineConfig.from_yaml(path)

    assert config.mannstein.min_straightness == 0.3**2, "min_r_score only existed in format 1"


def test_pipeline_config_reads_an_unversioned_file_as_the_current_format(tmp_path: Path):
    path = tmp_path / "new.yaml"
    path.write_text("mannstein:\n  min_straightness: 0.3\n")

    config = InstancePipelineConfig.from_yaml(path)

    assert config.mannstein.min_straightness == 0.3


@pytest.mark.parametrize(
    ("file", "new_value", "old_value"),
    [
        ("instance_pipeline.yaml", 0.006132138316667133, 0.07830797096507566),
        ("instance_pipeline_consensus.yaml", 0.0029922479572457735, 0.05470144383145452),
        ("mannstein_annotator_masks.yaml", 0.0029922479572457735, 0.05470144383145452),
    ],
)
def test_shipped_configs_reproduce_their_format_1_values(
    tmp_path: Path, file: str, new_value: float, old_value: float
) -> None:
    from pathlib import Path

    from deepcontrail.instance.config import MannsteinConfig

    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    text = (configs_dir / file).read_text()
    format_1 = text.replace("format_version: 2", "format_version: 1").replace(
        f"min_straightness: {new_value}", f"min_r_score: {old_value}"
    )
    assert "min_r_score" in format_1, "the format-1 text must carry the legacy key"

    format_2_path = tmp_path / "v2.yaml"
    format_2_path.write_text(text)
    format_1_path = tmp_path / "v1.yaml"
    format_1_path.write_text(format_1)

    loader = (
        MannsteinConfig.from_yaml
        if file == "mannstein_annotator_masks.yaml"
        else InstancePipelineConfig.from_yaml
    )

    assert loader(format_1_path) == loader(format_2_path), (
        f"configs/{file} loads the same from its format-1 and format-2 text"
    )


def test_configs_reject_unknown_fields_in_code():
    with pytest.raises(ValueError, match="grow_treshold"):
        InstancePipelineConfig(grow_treshold=0.9)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="num_kernel"):
        MannsteinConfig(num_kernel=24)  # type: ignore[call-arg]


def test_shipped_annotator_mask_config_is_the_consensus_presets_mannstein():
    from pathlib import Path

    shipped = Path(__file__).resolve().parents[1] / "configs" / "mannstein_annotator_masks.yaml"

    cfg = MannsteinConfig.from_yaml(shipped)

    assert cfg == InstancePipelineConfig.consensus().mannstein, (
        "one set of annotator-mask settings, shared with the consensus preset"
    )
    assert cfg.line_filter is False
