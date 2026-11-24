"""Tests for scripts/export_predictions.py — a bundle + split -> one map per scene."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _memory_bundle(tmp_path: Path) -> Path:
    """Build a track-memory bundle (native_size 8) whose memory channel changes the output.

    Like ``_bundle`` but with ``track_memory=True``; every memory projection's last conv weight is
    set to 0.05 (they start at zero), so feeding a frame the previous frame's map moves its map.
    """
    import torch

    from deepcontrail.core.false_color import FalseColorConfig
    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.pretrained import PretrainedConfig, save_pretrained

    model_cfg = ModelConfig(
        architecture="Unet", encoder_name="resnet18", encoder_weights=None, track_memory=True
    )
    config = PretrainedConfig(
        format_version=1,
        model=model_cfg,
        false_color=FalseColorConfig(**RECIPE),
        image_size=32,
        native_size=8,
        tta=True,
    )
    model = build_model(model_cfg)
    with torch.no_grad():
        for seq in model.encoder.inject:
            seq[-1].weight.copy_(torch.full_like(seq[-1].weight, 0.05))
    bundle = tmp_path / "bundle"
    save_pretrained(model, config, bundle)
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


def _export_track_memory(tmp_path: Path, synthetic_root: Path, out: Path) -> int:
    """Export a track-memory bundle over the two linked test scenes, no TTA, on the CPU."""
    _write_neighbours(synthetic_root)
    return _load_script().main(
        [
            "--bundle",
            str(_memory_bundle(tmp_path)),
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


def test_export_track_memory_feeds_each_map_to_the_next_frame(tmp_path: Path, synthetic_root: Path):
    """Each scene is fed the map of its previous frame; the first starts with an empty memory."""
    from deepcontrail.core.contract import read_band_manifest
    from deepcontrail.segmentation import load_pretrained, predict

    _write_neighbours(synthetic_root)
    bundle = _memory_bundle(tmp_path)
    out = tmp_path / "preds"

    assert (
        _load_script().main(
            [
                "--bundle",
                str(bundle),
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
        == 0
    )

    # Rebuild exactly what main did, frame by frame, and check each exported map matches.
    script = _load_script()
    data_cfg = script._select_data_config(
        _config(tmp_path, synthetic_root), {"data_root": None}, "test"
    )
    bands_dir = synthetic_root / "spectral_bands" / "test"
    manifest = read_band_manifest(synthetic_root)
    pretrained = load_pretrained(bundle, device="cpu")
    first_composite = script._build_composite(
        "test_000001", bands_dir, data_cfg, manifest, neighbours=None, context_k=0
    )
    second_composite = script._build_composite(
        "test_000002", bands_dir, data_cfg, manifest, neighbours=None, context_k=0
    )
    first = predict(pretrained, first_composite, tta=False, pixel_shift=data_cfg.pixel_shift)
    second = predict(
        pretrained, second_composite, memory=first, tta=False, pixel_shift=data_cfg.pixel_shift
    )

    np.testing.assert_allclose(
        np.load(out / "test_000001.npy"), first, atol=1e-6, err_msg="the first frame starts empty"
    )
    np.testing.assert_allclose(
        np.load(out / "test_000002.npy"), second, err_msg="the second frame gets the first's map"
    )
    no_memory = predict(pretrained, second_composite, tta=False, pixel_shift=data_cfg.pixel_shift)
    assert not np.allclose(second, no_memory), (
        "the second frame's map really used the memory of the first"
    )


def test_export_track_memory_without_neighbours_is_a_clear_error(
    tmp_path: Path, synthetic_root: Path
):
    """A track-memory bundle over a dataset without neighbours.json fails loudly, not silently."""
    from deepcontrail.core.contract import DatasetContractError

    out = tmp_path / "preds"

    with pytest.raises(DatasetContractError, match="neighbours"):
        _load_script().main(
            [
                "--bundle",
                str(_memory_bundle(tmp_path)),
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


def test_export_manifest_records_track_memory(tmp_path: Path, synthetic_root: Path):
    """The manifest records ``track_memory``: true for a memory bundle, false for a plain one."""
    out = tmp_path / "preds"
    _export_track_memory(tmp_path, synthetic_root, out)
    assert json.loads((out / "manifest.json").read_text())["track_memory"] is True

    plain = tmp_path / "preds_plain"
    _load_script().main(
        [
            "--bundle",
            str(_bundle(tmp_path)),
            "--config",
            str(_config(tmp_path, synthetic_root)),
            "--split",
            "test",
            "--out",
            str(plain),
            "--no-tta",
            "--device",
            "cpu",
        ]
    )
    assert json.loads((plain / "manifest.json").read_text())["track_memory"] is False


def test_in_time_order_follows_the_links():
    """Order sequences by their first scene, each following its next links; a broken link splits."""
    neighbours = {
        "a": (None, "b"),
        "b": ("a", "c"),
        "c": ("b", None),
        "x": (None, None),
    }
    script = _load_script()

    assert script._in_time_order(["b", "c", "a", "x"], neighbours) == ["a", "b", "c", "x"]
    assert script._in_time_order(["a", "c"], neighbours) == ["a", "c"], (
        "a selection that skips the middle frame is two sequences"
    )


def _second_frame_with_options(tmp_path: Path, root: Path, *options: str) -> tuple[Any, ...]:
    """Export the two-frame sequence with ``options``; return (exported second map, helpers)."""
    from deepcontrail.core.contract import read_band_manifest
    from deepcontrail.segmentation import load_pretrained

    _write_neighbours(root)
    bundle = _memory_bundle(tmp_path)
    out = tmp_path / "preds"
    args = ["--bundle", str(bundle), "--config", str(_config(tmp_path, root)), "--split", "test"]
    args += ["--out", str(out), "--no-tta", "--device", "cpu", *options]
    assert _load_script().main(args) == 0

    script = _load_script()
    data_cfg = script._select_data_config(_config(tmp_path, root), {"data_root": None}, "test")
    bands_dir = root / "spectral_bands" / "test"
    manifest = read_band_manifest(root)
    composites = [
        script._build_composite(s, bands_dir, data_cfg, manifest, neighbours=None, context_k=0)
        for s in ("test_000001", "test_000002")
    ]
    pretrained = load_pretrained(bundle, device="cpu")
    return np.load(out / "test_000002.npy"), pretrained, composites, data_cfg, out


def test_export_track_memory_empty_feeds_no_memory(tmp_path: Path, synthetic_root: Path):
    """--memory empty predicts every frame with an empty memory (the model without its memory)."""
    from deepcontrail.segmentation import predict

    exported, pretrained, (_, second), data_cfg, _ = _second_frame_with_options(
        tmp_path, synthetic_root, "--memory", "empty"
    )

    expected = predict(pretrained, second, tta=False, pixel_shift=data_cfg.pixel_shift)
    np.testing.assert_allclose(exported, expected, atol=1e-6, err_msg="no memory was fed")


def test_export_track_memory_threshold_and_scale_shape_the_fed_map(
    tmp_path: Path, synthetic_root: Path
):
    """The fed map keeps only values >= --memory-threshold, then is multiplied by --memory-scale."""
    from deepcontrail.segmentation import predict

    exported, pretrained, (first_c, second), data_cfg, out = _second_frame_with_options(
        tmp_path, synthetic_root, "--memory-threshold", "0.5", "--memory-scale", "0.5"
    )

    first = predict(pretrained, first_c, tta=False, pixel_shift=data_cfg.pixel_shift)
    fed = np.where(first >= 0.5, first, 0.0).astype(np.float32) * 0.5
    expected = predict(pretrained, second, memory=fed, tta=False, pixel_shift=data_cfg.pixel_shift)
    np.testing.assert_allclose(exported, expected, atol=1e-6, err_msg="thresholded, then scaled")
    recorded = json.loads((out / "manifest.json").read_text())
    assert (recorded["memory"], recorded["memory_threshold"], recorded["memory_scale"]) == (
        "feed",
        0.5,
        0.5,
    ), "the manifest records how the memory was fed"


def test_export_reverse_runs_each_sequence_backward_in_time(tmp_path: Path, synthetic_root: Path):
    """--reverse: the last frame starts with an empty memory and feeds the frame before it."""
    from deepcontrail.segmentation import predict

    exported_second, pretrained, (first_c, second_c), data_cfg, out = _second_frame_with_options(
        tmp_path, synthetic_root, "--reverse"
    )

    second = predict(pretrained, second_c, tta=False, pixel_shift=data_cfg.pixel_shift)
    first = predict(pretrained, first_c, memory=second, tta=False, pixel_shift=data_cfg.pixel_shift)
    np.testing.assert_allclose(exported_second, second, atol=1e-6, err_msg="last frame: empty")
    np.testing.assert_allclose(
        np.load(out / "test_000001.npy"), first, atol=1e-6, err_msg="fed the later frame's map"
    )
    assert json.loads((out / "manifest.json").read_text())["reverse"] is True


def test_export_memory_flows_warp_the_fed_map_onto_the_frame(tmp_path: Path, synthetic_root: Path):
    """--memory-flows: the fed map is warped by <stem>_prev.npy (a(y) ~ prev(y + f(y)))."""
    from deepcontrail.segmentation import predict

    flows = tmp_path / "flows"
    flows.mkdir()
    shift = np.zeros((16, 16, 2), np.float16)
    shift[..., 1] = 1.0  # every pixel of frame 2 comes from one row below in frame 1
    np.save(flows / "test_000002_prev.npy", shift)

    exported, pretrained, (first_c, second_c), data_cfg, _ = _second_frame_with_options(
        tmp_path, synthetic_root, "--memory-flows", str(flows)
    )

    first = predict(pretrained, first_c, tta=False, pixel_shift=data_cfg.pixel_shift)
    warped = np.zeros_like(first)
    warped[:-1] = first[1:]  # row r takes row r + 1; the last row has no source (0)
    expected = predict(
        pretrained, second_c, memory=warped, tta=False, pixel_shift=data_cfg.pixel_shift
    )
    np.testing.assert_allclose(exported, expected, atol=1e-5, err_msg="the memory was warped")
