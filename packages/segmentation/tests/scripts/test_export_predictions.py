"""Tests for scripts/export_predictions.py — a bundle + split -> one map per scene."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_predictions.py"
RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_predictions", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(tmp_path: Path) -> Path:
    """Build a tiny pretrained bundle (native_size 8) with random weights, no network."""
    from deepcontrail.core.false_color import FalseColorConfig
    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.pretrained import PretrainedConfig, save_pretrained

    model_cfg = ModelConfig(architecture="Unet", encoder_name="resnet18", encoder_weights=None)
    config = PretrainedConfig(
        format_version=1,
        model=model_cfg,
        false_color=FalseColorConfig(**RECIPE),
        image_size=32,
        native_size=8,
        tta=True,
    )
    bundle = tmp_path / "bundle"
    save_pretrained(build_model(model_cfg), config, bundle)
    return bundle


def _config(tmp_path: Path, root: Path) -> Path:
    """Write a training YAML whose data section points at the synthetic dataset."""
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"root": str(root), "false_color": RECIPE, "image_size": 32},
                "model": {
                    "architecture": "Unet",
                    "encoder_name": "resnet18",
                    "encoder_weights": None,
                },
            }
        )
    )
    return path


def test_export_writes_a_float32_map_per_test_scene(tmp_path: Path, synthetic_root: Path):
    """One (16,16) float32 map in [0,1] per test scene, from the tiled predict path."""
    out = tmp_path / "preds"

    code = _load_script().main(
        [
            "--bundle",
            str(_bundle(tmp_path)),
            "--config",
            str(_config(tmp_path, synthetic_root)),
            "--split",
            "test",
            "--out",
            str(out),
            "--no-tta",
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    maps = sorted(p.name for p in out.glob("*.npy"))
    assert maps == ["test_000001.npy", "test_000002.npy"], "one map per test scene"
    for path in out.glob("*.npy"):
        arr = np.load(path)
        assert arr.shape == (16, 16)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0, "sigmoid probabilities stay >= 0"
        assert arr.max() <= 1.0, "sigmoid probabilities stay <= 1"


def test_export_manifest_records_the_split_and_effective_tta(tmp_path: Path, synthetic_root: Path):
    """The manifest records the run settings, with TTA off as asked."""
    out = tmp_path / "preds"

    _load_script().main(
        [
            "--bundle",
            str(_bundle(tmp_path)),
            "--config",
            str(_config(tmp_path, synthetic_root)),
            "--split",
            "test",
            "--out",
            str(out),
            "--no-tta",
            "--device",
            "cpu",
        ]
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["split"] == "test"
    assert manifest["n_scenes"] == 2
    assert manifest["tta"] is False, "--no-tta must be recorded as the effective setting"
    assert manifest["created"], "a creation timestamp is written"


def test_export_skips_maps_already_written_on_rerun(tmp_path: Path, synthetic_root: Path):
    """An existing map is left untouched, so a rerun resumes instead of redoing work."""
    out = tmp_path / "preds"
    script = _load_script()
    common = [
        "--bundle",
        str(_bundle(tmp_path)),
        "--config",
        str(_config(tmp_path, synthetic_root)),
        "--split",
        "test",
        "--out",
        str(out),
        "--no-tta",
        "--device",
        "cpu",
    ]
    script.main(common)

    untouched = out / "test_000001.npy"
    sentinel = np.full((16, 16), 0.123, np.float32)
    np.save(untouched, sentinel)

    script.main(common)

    assert np.array_equal(np.load(untouched), sentinel), "an existing map must not be rewritten"


def test_export_limit_keeps_only_the_first_scenes(tmp_path: Path, synthetic_root: Path):
    """--limit caps the number of exported maps (smoke tests)."""
    out = tmp_path / "preds"

    _load_script().main(
        [
            "--bundle",
            str(_bundle(tmp_path)),
            "--config",
            str(_config(tmp_path, synthetic_root)),
            "--split",
            "test",
            "--out",
            str(out),
            "--no-tta",
            "--device",
            "cpu",
            "--limit",
            "1",
        ]
    )

    assert sorted(p.name for p in out.glob("*.npy")) == ["test_000001.npy"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_scenes"] == 1


def _write_neighbours(root: Path) -> None:
    """Link the two test scenes into one sequence, so k=1 resolves to existing frames."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "neighbours.json").write_text(
        json.dumps(
            {
                "test_000001": {"prev": None, "next": "test_000002"},
                "test_000002": {"prev": "test_000001", "next": None},
            }
        )
    )


def _context_bundle(tmp_path: Path) -> Path:
    """Build a k=1 bundle (9 input channels) with random weights, no network."""
    from deepcontrail.core.false_color import FalseColorConfig
    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.pretrained import PretrainedConfig, save_pretrained

    model_cfg = ModelConfig(
        architecture="Unet", encoder_name="resnet18", encoder_weights=None, in_channels=9
    )
    config = PretrainedConfig(
        format_version=1,
        model=model_cfg,
        false_color=FalseColorConfig(**RECIPE),
        image_size=32,
        native_size=8,
        tta=True,
        context_frames=1,
    )
    bundle = tmp_path / "bundle"
    save_pretrained(build_model(model_cfg), config, bundle)
    return bundle


def test_export_stacks_context_frames_with_a_temporal_bundle(tmp_path: Path, synthetic_root: Path):
    """A k=1 bundle reads neighbours and stacks the 2k+1 frames into a 9-channel composite."""
    _write_neighbours(synthetic_root)
    out = tmp_path / "preds"

    code = _load_script().main(
        [
            "--bundle",
            str(_context_bundle(tmp_path)),
            "--config",
            str(_config(tmp_path, synthetic_root)),
            "--split",
            "test",
            "--out",
            str(out),
            "--no-tta",
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    maps = sorted(p.name for p in out.glob("*.npy"))
    assert maps == ["test_000001.npy", "test_000002.npy"], "one map per test scene"
    for path in out.glob("*.npy"):
        arr = np.load(path)
        assert arr.shape == (16, 16)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0


def test_export_temporal_bundle_without_neighbours_is_a_clear_error(
    tmp_path: Path, synthetic_root: Path
):
    """A k=1 bundle over a dataset without neighbours.json fails loudly, not silently."""
    from deepcontrail.core.contract import DatasetContractError

    out = tmp_path / "preds"

    with pytest.raises(DatasetContractError, match="neighbours"):
        _load_script().main(
            [
                "--bundle",
                str(_context_bundle(tmp_path)),
                "--config",
                str(_config(tmp_path, synthetic_root)),
                "--split",
                "test",
                "--out",
                str(out),
                "--no-tta",
                "--device",
                "cpu",
            ]
        )
