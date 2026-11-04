"""Tests for deepcontrail.segmentation.pretrained — weight bundles, loading and prediction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


def _config(**overrides: object):  # noqa: ANN202
    from deepcontrail.segmentation.pretrained import PretrainedConfig

    return PretrainedConfig.model_validate(
        {
            "format_version": 1,
            "model": {
                "architecture": "Unet",
                "encoder_name": "resnet18",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 1,
            },
            "false_color": RECIPE,
            "image_size": 32,
            "native_size": 16,
        }
        | overrides
    )


def _model(seed: int = 0) -> torch.nn.Module:
    from deepcontrail.segmentation.models import build_model

    torch.manual_seed(seed)
    return build_model(_config().model).eval()


def _composite(n: int | None = None, seed: int = 0) -> np.ndarray:
    shape = (16, 16, 3) if n is None else (n, 16, 16, 3)
    return np.random.RandomState(seed).uniform(0, 1, shape).astype(np.float32)


# --- bundle on disk ---------------------------------------------------------------------


def test_save_pretrained_writes_weights_and_config(tmp_path: Path):
    from deepcontrail.segmentation.pretrained import save_pretrained

    save_pretrained(_model(), _config(), tmp_path)

    assert (tmp_path / "model.safetensors").is_file()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["format_version"] == 1
    assert config["false_color"]["channels"][1]["bands"] == ["ir_105", "ir_87"]


def test_load_pretrained_round_trip_gives_identical_outputs(tmp_path: Path):
    from deepcontrail.segmentation.pretrained import load_pretrained, save_pretrained

    model = _model()
    save_pretrained(model, _config(), tmp_path)
    x = torch.randn(2, 3, 32, 32)

    loaded = load_pretrained(tmp_path)

    with torch.no_grad():
        assert torch.equal(loaded.model(x), model(x)), "the bundle must hold the exact weights"
    assert loaded.config == _config()


def test_load_pretrained_never_downloads_encoder_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from deepcontrail.segmentation import pretrained

    save_pretrained = pretrained.save_pretrained
    save_pretrained(
        _model(),
        _config(model=_config().model.model_copy(update={"encoder_weights": "imagenet"})),
        tmp_path,
    )
    built = []
    real_build = pretrained.build_model
    monkeypatch.setattr(pretrained, "build_model", lambda cfg: built.append(cfg) or real_build(cfg))

    pretrained.load_pretrained(tmp_path)

    assert built[0].encoder_weights is None, "weights come from model.safetensors"


def test_load_pretrained_from_the_hub_downloads_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import huggingface_hub

    from deepcontrail.segmentation.pretrained import load_pretrained, save_pretrained

    save_pretrained(_model(), _config(), tmp_path)
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    loaded = load_pretrained("eurocontrol-asu/some-model", revision="v0.1.0")

    assert calls[0]["repo_id"] == "eurocontrol-asu/some-model"
    assert calls[0]["revision"] == "v0.1.0"
    assert loaded.config.image_size == 32


def test_load_pretrained_exposes_the_bundle_folder(tmp_path: Path):
    from deepcontrail.segmentation.pretrained import load_pretrained, save_pretrained

    save_pretrained(_model(), _config(), tmp_path)

    loaded = load_pretrained(tmp_path)

    assert loaded.directory == tmp_path, "optional files of the bundle are read from this folder"


def test_load_pretrained_from_the_hub_also_fetches_the_optional_instance_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import huggingface_hub

    from deepcontrail.segmentation.pretrained import load_pretrained, save_pretrained

    save_pretrained(_model(), _config(), tmp_path)
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    loaded = load_pretrained("eurocontrol-asu/some-model")

    patterns = set(calls[0]["allow_patterns"])
    assert {"instance_pipeline.yaml", "score_model.json"} <= patterns, (
        "a bundle may carry the instance step; downloading only the network would drop it"
    )
    assert loaded.directory == tmp_path


def test_load_pretrained_missing_bundle_file_is_a_clear_error(tmp_path: Path):
    from deepcontrail.segmentation.pretrained import load_pretrained, save_pretrained

    save_pretrained(_model(), _config(), tmp_path)
    (tmp_path / "model.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        load_pretrained(tmp_path)


def test_pretrained_config_rejects_unknown_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _config(epochs=40)


# --- predict ----------------------------------------------------------------------------


def test_predict_returns_probabilities_at_the_input_size():
    from deepcontrail.segmentation.pretrained import Pretrained, predict

    probs = predict(Pretrained(_model(), _config()), _composite())

    assert probs.shape == (16, 16), "probabilities come back at the composite's resolution"
    assert probs.dtype == np.float32
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_predict_accepts_a_batch():
    from deepcontrail.segmentation.pretrained import Pretrained, predict

    pretrained = Pretrained(_model(), _config())
    batch = _composite(n=3)

    probs = predict(pretrained, batch, batch_size=2)

    assert probs.shape == (3, 16, 16)
    np.testing.assert_allclose(probs[1], predict(pretrained, batch[1]), atol=1e-6)


def test_predict_rejects_a_composite_that_is_not_three_channels():
    from deepcontrail.segmentation.pretrained import Pretrained, predict

    with pytest.raises(ValueError, match="3"):
        predict(Pretrained(_model(), _config()), np.zeros((16, 16, 10), np.float32))


@pytest.mark.parametrize("shape", [(32, 32, 3), (16, 24, 3), (2, 8, 8, 3)])
def test_predict_rejects_scenes_of_another_size_than_the_training_scenes(shape):
    from deepcontrail.segmentation.pretrained import Pretrained, predict

    composite = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="16 x 16"):
        predict(Pretrained(_model(), _config()), composite)


def test_predict_matches_the_training_pipeline(synthetic_root: Path):
    """Same composite, resize, normalisation and output resize as SegmentationDataset."""
    from deepcontrail.core.false_color import false_color_from_spectral
    from deepcontrail.segmentation import SegmentationDataConfig, SegmentationDataset
    from deepcontrail.segmentation.losses import LossConfig, build_loss
    from deepcontrail.segmentation.pretrained import Pretrained, predict
    from deepcontrail.segmentation.training import SegmentationModule, tta_predict

    data_cfg = SegmentationDataConfig(
        root=str(synthetic_root), split="test", false_color=RECIPE, image_size=32
    )
    dataset = SegmentationDataset(data_cfg)
    model = _model()
    module = SegmentationModule(
        model=model,
        loss_fn=build_loss(LossConfig(name="bce")),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=16,
    ).eval()
    image, _ = dataset[0]
    bands = np.load(synthetic_root / "spectral_bands" / "test" / f"{dataset.stems[0]}.npy")
    composite = false_color_from_spectral(bands, data_cfg.false_color, dataset.manifest)
    pretrained = Pretrained(model, _config())

    with torch.no_grad():
        expected = torch.sigmoid(module(image[None]))[0, 0].numpy()
        expected_tta = tta_predict(module, image[None])[0].numpy()

    np.testing.assert_allclose(predict(pretrained, composite), expected, atol=1e-6)
    np.testing.assert_allclose(predict(pretrained, composite, tta=True), expected_tta, atol=1e-6)


def test_predict_applies_the_pixel_shift():
    from deepcontrail.core.readers import shift_image
    from deepcontrail.segmentation.pretrained import Pretrained, predict

    pretrained = Pretrained(_model(), _config())
    composite = _composite()

    shifted = predict(pretrained, composite, pixel_shift=0.5)

    np.testing.assert_allclose(shifted, predict(pretrained, shift_image(composite, 0.5)), atol=1e-6)
    assert not np.allclose(shifted, predict(pretrained, composite))


def test_segmentation_package_exports_the_pretrained_api():
    import deepcontrail.segmentation as seg

    assert callable(seg.load_pretrained)
    assert callable(seg.predict)
