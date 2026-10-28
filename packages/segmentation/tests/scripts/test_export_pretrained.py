"""Tests for scripts/export_pretrained.py — Lightning checkpoint -> pretrained bundle."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_pretrained.py"
RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_pretrained", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(tmp_path: Path, root: Path) -> tuple[Path, Path, torch.nn.Module]:
    """Save a tiny trained-model checkpoint, as Lightning does, and its experiment YAML."""
    from deepcontrail.segmentation.losses import LossConfig, build_loss
    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.training import SegmentationModule

    model_cfg = {"architecture": "Unet", "encoder_name": "resnet18", "encoder_weights": None}
    torch.manual_seed(0)
    module = SegmentationModule(
        model=build_model(ModelConfig(**model_cfg)),
        loss_fn=build_loss(LossConfig(name="bce")),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=16,
    )
    ckpt = tmp_path / "best-val_dice=0.5000.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, ckpt)
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {"root": str(root), "false_color": RECIPE, "image_size": 32},
                "model": model_cfg,
            }
        )
    )
    return ckpt, config, module.model.eval()


def test_export_writes_a_bundle_with_the_checkpoint_weights(tmp_path: Path, synthetic_root: Path):
    from deepcontrail.segmentation.pretrained import load_pretrained

    ckpt, config, model = _checkpoint(tmp_path, synthetic_root)
    out = tmp_path / "bundle"

    code = _load_script().main(
        ["--checkpoint", str(ckpt), "--config", str(config), "--output-dir", str(out)]
    )

    assert code == 0
    loaded = load_pretrained(out)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(loaded.model(x), model(x)), (
            "the bundle must hold the checkpoint's weights"
        )
    assert loaded.config.native_size == 16, "native size comes from the checkpoint"
    assert loaded.config.image_size == 32
    assert loaded.config.false_color.channels[2].bands == ("ir_105",)


def test_export_records_metrics_and_provenance(tmp_path: Path, synthetic_root: Path):
    from deepcontrail.segmentation.pretrained import load_pretrained

    ckpt, config, _ = _checkpoint(tmp_path, synthetic_root)
    out = tmp_path / "bundle"

    _load_script().main(
        [
            "--checkpoint", str(ckpt), "--config", str(config), "--output-dir", str(out),
            "--metric", "test_dice_tta_at_0.5=0.696", "--run-name", "my-run",
        ]
    )  # fmt: skip

    loaded = load_pretrained(out)
    assert loaded.config.metrics == {"test_dice_tta_at_0.5": 0.696}
    assert loaded.config.provenance["run_name"] == "my-run"
    assert loaded.config.provenance["checkpoint"] == ckpt.name
    assert "deepcontrail_version" in loaded.config.provenance


def test_export_writes_a_draft_model_card(tmp_path: Path, synthetic_root: Path):
    ckpt, config, _ = _checkpoint(tmp_path, synthetic_root)
    out = tmp_path / "bundle"

    _load_script().main(
        [
            "--checkpoint", str(ckpt), "--config", str(config), "--output-dir", str(out),
            "--metric", "test_dice_tta_at_0.5=0.696",
        ]
    )  # fmt: skip

    card = (out / "README.md").read_text()
    assert "load_pretrained" in card, "the card shows how to use the weights"
    assert "test_dice_tta_at_0.5" in card
    assert "ir_123 - ir_105" in card, "the card states the composite recipe"
    assert "TO FILL IN" in card, "licence and attribution are decided by a person"
