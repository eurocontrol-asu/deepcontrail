"""Tests for deepcontrail.segmentation.training.run — the training entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deepcontrail.segmentation.models import ModelConfig

RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


def _config(tmp_path: Path, root: Path, **training: object) -> Path:
    """Write a tiny experiment: resnet18 U-Net at 32 px, one CPU-friendly epoch."""
    config = {
        "data": {
            "root": str(root),
            "false_color": RECIPE,
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
            "val_fraction": 0.2,
        },
        "model": {
            "architecture": "Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        },
        "loss": {"name": "bce"},
        "training": {"epochs": 1, "precision": "32-true", **training},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


# --- end to end -----------------------------------------------------------------------


def test_train_with_the_csv_logger_writes_metrics_and_a_best_checkpoint(
    tmp_path: Path, synthetic_root: Path
):
    import torch

    from deepcontrail.segmentation.training import train

    torch.manual_seed(0)
    config = _config(tmp_path, synthetic_root)

    run = train(config, output_dir=tmp_path / "runs", run_name="tiny", accelerator="cpu")

    assert run.run_dir == tmp_path / "runs" / "tiny"
    assert run.best_checkpoint is not None, "the best checkpoint by val_dice must be saved"
    assert run.best_checkpoint.parent == run.run_dir
    assert (run.run_dir / "metrics.csv").is_file(), "the default csv logger writes metrics.csv"
    assert run.best_val_dice is not None


def test_segmentation_package_exports_train():
    from deepcontrail.segmentation import train
    from deepcontrail.segmentation.training.run import train as run_train

    assert train is run_train


# --- configuration --------------------------------------------------------------------


def test_training_config_logger_defaults_to_csv():
    from deepcontrail.segmentation.training import TrainingConfig

    assert TrainingConfig().logger == "csv"


def test_training_config_rejects_an_unknown_logger():
    from pydantic import ValidationError

    from deepcontrail.segmentation.training import TrainingConfig

    with pytest.raises(ValidationError, match="logger"):
        TrainingConfig(logger="tensorboard")  # type: ignore[arg-type]


def test_prepare_training_logger_argument_overrides_the_config(
    tmp_path: Path, synthetic_root: Path
):
    pytest.importorskip("mlflow", reason="needs deepcontrail-segmentation[mlflow]")
    from pytorch_lightning.loggers import MLFlowLogger

    from deepcontrail.segmentation.training import prepare_training

    config = _config(tmp_path, synthetic_root, logger="csv")

    run = prepare_training(
        config,
        output_dir=tmp_path / "runs",
        run_name="x",
        logger="mlflow",
        mlflow_uri=f"sqlite:///{tmp_path}/mlflow.db",
        accelerator="cpu",
    )

    assert isinstance(run.trainer.logger, MLFlowLogger)
    assert run.training_cfg.logger == "mlflow", "the run config must record the logger used"


def test_prepare_training_applies_overrides(tmp_path: Path, synthetic_root: Path):
    from deepcontrail.segmentation.training import prepare_training

    config = _config(tmp_path, synthetic_root)

    run = prepare_training(
        config, output_dir=tmp_path / "runs", overrides={"epochs": 3}, accelerator="cpu"
    )

    assert run.training_cfg.epochs == 3
    assert run.trainer.max_epochs == 3


def test_load_configs_accepts_a_mapping_or_no_overrides(tmp_path: Path, synthetic_root: Path):
    from deepcontrail.segmentation.training import load_configs

    config = _config(tmp_path, synthetic_root)

    plain = load_configs(config)
    overridden = load_configs(config, {"data_root": "/elsewhere", "epochs": 7})

    assert plain[0].root == str(synthetic_root)
    assert overridden[0].root == "/elsewhere"
    assert overridden[5].epochs == 7


def test_default_run_name_names_the_model_and_time():
    from deepcontrail.segmentation.models import ModelConfig
    from deepcontrail.segmentation.training import default_run_name

    name = default_run_name(ModelConfig(architecture="Unet", encoder_name="resnet18"))

    assert name.startswith("Unet-resnet18-")


# --- initial weights (fine-tuning) -----------------------------------------------------


def _tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        architecture="Unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def _save_bundle(tmp_path: Path, model_cfg, *, native_size: int = 16) -> Path:
    """Save a pretrained bundle holding a freshly built model's weights."""
    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.pretrained import PretrainedConfig, save_pretrained

    weights = build_model(model_cfg)
    config = PretrainedConfig(
        format_version=1,
        model=model_cfg,
        false_color=RECIPE,
        image_size=32,
        native_size=native_size,
    )
    directory = tmp_path / "bundle"
    save_pretrained(weights, config, directory)
    return directory, weights


def test_load_initial_weights_from_bundle_copies_every_weight(tmp_path: Path) -> None:
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    model_cfg = _tiny_model_cfg()
    directory, source = _save_bundle(tmp_path, model_cfg)
    target = build_model(model_cfg)

    load_initial_weights(target, str(directory))

    after = target.state_dict()
    for key, tensor in source.state_dict().items():
        assert torch.equal(after[key], tensor), f"weight {key} must come from the bundle"


def test_load_initial_weights_from_ckpt_strips_the_model_prefix(tmp_path: Path) -> None:
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    model_cfg = _tiny_model_cfg()
    source = build_model(model_cfg)
    # A Lightning checkpoint keeps the module's weights under a "model." prefix.
    state_dict = {f"model.{key}": value for key, value in source.state_dict().items()}
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": state_dict}, checkpoint)
    target = build_model(model_cfg)

    load_initial_weights(target, str(checkpoint))

    after = target.state_dict()
    for key, value in state_dict.items():
        assert torch.equal(after[key.removeprefix("model.")], value), (
            "the 'model.' prefix must be stripped before loading the checkpoint's weights"
        )


def test_load_initial_weights_with_other_architecture_raises(tmp_path: Path) -> None:
    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    model_cfg = _tiny_model_cfg()
    directory, _source = _save_bundle(tmp_path, model_cfg)
    target = build_model(model_cfg.model_copy(update={"classes": 2}))

    with pytest.raises(RuntimeError, match=r"size mismatch|Unexpected key|Missing key"):
        load_initial_weights(target, str(directory))


def test_prepare_training_with_init_from_starts_from_those_weights(
    tmp_path: Path, synthetic_root: Path
) -> None:
    import torch

    from deepcontrail.segmentation.training import prepare_training

    model_cfg = _tiny_model_cfg()
    directory, source = _save_bundle(tmp_path, model_cfg)
    config = _config(tmp_path, synthetic_root, init_from=str(directory))

    run = prepare_training(config, output_dir=tmp_path / "runs", run_name="ft", accelerator="cpu")

    after = run.module.model.state_dict()
    for key, tensor in source.state_dict().items():
        assert torch.equal(after[key], tensor), f"the module must start from the bundle's {key}"


def test_load_initial_weights_widens_input_convs_on_the_centre_frame(tmp_path: Path) -> None:
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    source_cfg = _tiny_model_cfg().model_copy(update={"in_channels": 3})
    target_cfg = _tiny_model_cfg().model_copy(update={"in_channels": 9})
    directory, source = _save_bundle(tmp_path, source_cfg)
    target = build_model(target_cfg)

    load_initial_weights(target, str(directory))

    after = target.state_dict()
    for key, src in source.state_dict().items():
        tgt = after[key]
        if src.shape == tgt.shape:
            assert torch.equal(tgt, src), f"non-temporal weight {key} must come from the bundle"
        else:
            # A widened input conv: the source on the centre block, exactly zero elsewhere.
            m = tgt.shape[1] // src.shape[1]
            half = m // 2
            source_in = src.shape[1]
            assert torch.equal(tgt[:, half * source_in : (half + 1) * source_in], src), (
                f"the centre block of {key} must be the source weights"
            )
            left = tgt[:, : half * source_in]
            right = tgt[:, (half + 1) * source_in :]
            assert torch.count_nonzero(torch.cat([left, right], dim=1)) == 0, (
                f"the neighbour frames of {key} must start at zero"
            )


def test_widened_model_gives_the_source_output_on_the_centre_frame(tmp_path: Path) -> None:
    import numpy as np
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    source_cfg = _tiny_model_cfg().model_copy(update={"in_channels": 3})
    target_cfg = _tiny_model_cfg().model_copy(update={"in_channels": 9})
    directory, source = _save_bundle(tmp_path, source_cfg)
    target = build_model(target_cfg)

    load_initial_weights(target, str(directory))

    # The centre frame is the same image whatever the two neighbour frames are: with the neighbour
    # blocks zero-initialised, the stem conv — and so the whole network — sees only the centre.
    centre = np.random.RandomState(0).uniform(0, 1, (16, 16, 3)).astype(np.float32)
    stacked = np.concatenate(
        [
            np.random.RandomState(1).uniform(0, 1, (16, 16, 3)).astype(np.float32),
            centre,
            np.random.RandomState(2).uniform(0, 1, (16, 16, 3)).astype(np.float32),
        ],
        axis=-1,
    )
    source.eval()
    target.eval()
    centre_t = torch.from_numpy(centre).permute(2, 0, 1)[None]
    stacked_t = torch.from_numpy(stacked).permute(2, 0, 1)[None]
    with torch.no_grad():
        from_source = torch.sigmoid(source(centre_t))
        from_target = torch.sigmoid(target(stacked_t))
    np.testing.assert_allclose(
        from_target.numpy(),
        from_source.numpy(),
        atol=1e-5,
        err_msg=(
            "a 9-channel model initialised on the centre frame must match the 3-channel source on it"
        ),
    )


def test_load_initial_weights_other_mismatch_still_raises(tmp_path: Path) -> None:
    """An even channel multiple is not a temporal widening, so strict loading still rejects it."""
    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    source_cfg = _tiny_model_cfg().model_copy(update={"in_channels": 3})
    directory, _source = _save_bundle(tmp_path, source_cfg)
    target = build_model(_tiny_model_cfg().model_copy(update={"in_channels": 6}))

    with pytest.raises(RuntimeError, match=r"size mismatch|Unexpected key|Missing key"):
        load_initial_weights(target, str(directory))


def test_prepare_training_rejects_in_channels_not_matching_context(
    tmp_path: Path, synthetic_root: Path
) -> None:
    """context_frames=1 needs in_channels=9; a 3-channel model is rejected before anything builds."""
    import yaml

    from deepcontrail.segmentation.training import prepare_training

    config = {
        "data": {
            "root": str(synthetic_root),
            "false_color": RECIPE,
            "image_size": 32,
            "context_frames": 1,
        },
        "model": {
            "architecture": "Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
        },
        "loss": {"name": "bce"},
        "training": {"epochs": 1, "precision": "32-true"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="9"):
        prepare_training(path, output_dir=tmp_path / "runs", accelerator="cpu")


def test_prepare_training_rejects_a_non_three_channel_model_without_context(
    tmp_path: Path, synthetic_root: Path
) -> None:
    """context_frames=0 needs in_channels=3; a 6-channel model is rejected with both numbers."""
    import yaml

    from deepcontrail.segmentation.training import prepare_training

    config = {
        "data": {"root": str(synthetic_root), "false_color": RECIPE, "image_size": 32},
        "model": {
            "architecture": "Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 6,
            "classes": 1,
        },
        "loss": {"name": "bce"},
        "training": {"epochs": 1, "precision": "32-true"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="3"):
        prepare_training(path, output_dir=tmp_path / "runs", accelerator="cpu")


# --- gradient checkpointing ------------------------------------------------------------


def test_enable_gradient_checkpointing_switches_on_every_timm_stage():
    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.training.run import enable_gradient_checkpointing

    model = build_model(
        ModelConfig(
            architecture="Unet",
            encoder_name="tu-convnext_atto",
            encoder_weights=None,
        )
    )

    count = enable_gradient_checkpointing(model)

    stages = [
        module
        for module in model.modules()
        if hasattr(module, "blocks")
        and isinstance(getattr(module, "grad_checkpointing", None), bool)
    ]
    assert count == len(stages), "the returned count must be the number of stages switched on"
    assert count > 0, "a timm encoder must have at least one stage that supports checkpointing"
    for stage in stages:
        assert stage.grad_checkpointing is True, "every timm stage must be switched on"
    wrappers = [
        module
        for module in model.modules()
        if not hasattr(module, "blocks")
        and isinstance(getattr(module, "grad_checkpointing", None), bool)
    ]
    assert wrappers, "the timm feature wrapper (a flag but no blocks) should exist here"
    for wrapper in wrappers:
        assert wrapper.grad_checkpointing is False, (
            "the feature wrapper has the flag but no blocks, so it must be left off"
        )


def test_gradient_checkpointing_gives_the_same_gradients():
    import copy

    import torch

    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.training.run import enable_gradient_checkpointing

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(
            architecture="Unet",
            encoder_name="tu-convnext_atto",
            encoder_weights=None,
        )
    )
    checkpointed = copy.deepcopy(model)
    enable_gradient_checkpointing(checkpointed)

    model.train()
    checkpointed.train()
    x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))
    model(x).mean().backward()
    checkpointed(x).mean().backward()

    for name, param in model.named_parameters():
        plain_grad = param.grad
        ckpt_grad = dict(checkpointed.named_parameters())[name].grad
        assert plain_grad is not None, f"the plain model must have a gradient for {name}"
        assert ckpt_grad is not None, f"the checkpointed model must have a gradient for {name}"
        assert torch.allclose(plain_grad, ckpt_grad, atol=1e-6), (
            f"the gradients of {name} must be the same with and without checkpointing"
        )


def test_enable_gradient_checkpointing_without_timm_stages_raises():
    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import enable_gradient_checkpointing

    model = build_model(_tiny_model_cfg())

    with pytest.raises(ValueError, match="gradient_checkpointing"):
        enable_gradient_checkpointing(model)


def test_prepare_training_with_gradient_checkpointing_switches_it_on(
    tmp_path: Path, synthetic_root: Path
):
    from deepcontrail.segmentation.training import prepare_training

    config = _config(tmp_path, synthetic_root, gradient_checkpointing=True)
    with config.open() as f:
        data = yaml.safe_load(f)
    data["model"]["encoder_name"] = "tu-convnext_atto"
    with config.open("w") as f:
        yaml.safe_dump(data, f)

    run = prepare_training(config, output_dir=tmp_path / "runs", run_name="gc", accelerator="cpu")

    stages = [
        module
        for module in run.module.model.modules()
        if hasattr(module, "blocks")
        and isinstance(getattr(module, "grad_checkpointing", None), bool)
    ]
    assert stages, "a timm encoder must have at least one stage that supports checkpointing"
    for stage in stages:
        assert stage.grad_checkpointing is True, "prepare_training must switch checkpointing on"


def test_prepare_training_gradient_checkpointing_with_resnet_raises(
    tmp_path: Path, synthetic_root: Path
):
    from deepcontrail.segmentation.training import prepare_training

    config = _config(tmp_path, synthetic_root, gradient_checkpointing=True)

    with pytest.raises(ValueError, match="gradient_checkpointing"):
        prepare_training(config, output_dir=tmp_path / "runs", run_name="gc", accelerator="cpu")


# --- loggers ---------------------------------------------------------------------------


def test_build_logger_csv_writes_into_the_run_directory(tmp_path: Path):
    from pytorch_lightning.loggers import CSVLogger

    from deepcontrail.segmentation.training import build_logger

    logger = build_logger("csv", output_dir=tmp_path, run_name="r1")

    assert isinstance(logger, CSVLogger)
    assert Path(logger.log_dir) == tmp_path / "r1"


def test_build_logger_without_its_package_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from deepcontrail.segmentation.training import build_logger, run

    monkeypatch.setattr(run, "_module_available", lambda name: False)

    with pytest.raises(ImportError, match=r"deepcontrail-segmentation\[mlflow\]"):
        build_logger("mlflow", output_dir=tmp_path, run_name="r1")


def test_build_logger_mlflow_uses_the_given_uri(tmp_path: Path):
    pytest.importorskip("mlflow", reason="needs deepcontrail-segmentation[mlflow]")
    from deepcontrail.segmentation.training import build_logger

    uri = f"sqlite:///{tmp_path}/given.db"

    logger = build_logger("mlflow", output_dir=tmp_path, run_name="r1", mlflow_uri=uri)

    assert logger._tracking_uri == uri  # type: ignore[attr-defined]


def test_build_logger_mlflow_falls_back_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("mlflow", reason="needs deepcontrail-segmentation[mlflow]")
    from deepcontrail.segmentation.training import build_logger

    uri = f"sqlite:///{tmp_path}/env.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    logger = build_logger("mlflow", output_dir=tmp_path, run_name="r1")

    assert logger._tracking_uri == uri  # type: ignore[attr-defined]


def test_build_logger_mlflow_defaults_to_a_database_in_the_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("mlflow", reason="needs deepcontrail-segmentation[mlflow]")
    from deepcontrail.segmentation.training import build_logger

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    logger = build_logger("mlflow", output_dir=tmp_path, run_name="r1")

    assert logger._tracking_uri == f"sqlite:///{tmp_path.resolve()}/mlflow.db"  # type: ignore[attr-defined]


# --- command line ----------------------------------------------------------------------------


def test_cli_passes_its_arguments_to_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from deepcontrail.segmentation.training import run

    calls: list[tuple[object, dict]] = []

    class _Done:
        best_checkpoint = tmp_path / "best.ckpt"
        best_val_dice = 0.5
        run_dir = tmp_path

    def fake_train(config_path: object, **kwargs: object) -> _Done:
        calls.append((config_path, kwargs))
        return _Done()

    monkeypatch.setattr(run, "train", fake_train)

    code = run.main(["exp.yaml", "--output-dir", "out", "--logger", "mlflow", "--epochs", "2"])

    assert code == 0
    (config_path, kwargs) = calls[0]
    assert str(config_path) == "exp.yaml"
    assert kwargs["logger"] == "mlflow"
    assert kwargs["overrides"] == {"epochs": 2}
    assert "best.ckpt" in capsys.readouterr().out


def test_prepare_training_uses_text_progress_and_summary_even_with_rich(
    tmp_path: Path, synthetic_root: Path
):
    from pytorch_lightning.callbacks import (
        ModelSummary,
        RichModelSummary,
        RichProgressBar,
        TQDMProgressBar,
    )

    from deepcontrail.segmentation.training import prepare_training

    pytest.importorskip("rich")  # Lightning would pick its Rich bar, silent in log files
    run = prepare_training(
        _config(tmp_path, synthetic_root), output_dir=tmp_path, accelerator="cpu"
    )

    callbacks = run.trainer.callbacks
    assert any(isinstance(c, TQDMProgressBar) for c in callbacks), "nohup logs need a text bar"
    assert not any(isinstance(c, (RichProgressBar, RichModelSummary)) for c in callbacks)
    assert any(type(c) is ModelSummary for c in callbacks)


def test_prepare_training_without_override_keeps_the_config_logger(
    tmp_path: Path, synthetic_root: Path
):
    from pytorch_lightning.loggers import CSVLogger

    from deepcontrail.segmentation.training import prepare_training

    run = prepare_training(
        _config(tmp_path, synthetic_root), output_dir=tmp_path, accelerator="cpu"
    )

    assert isinstance(run.trainer.logger, CSVLogger)
    assert run.training_cfg.logger == "csv"


def test_prepare_training_monitors_the_learning_rate(tmp_path: Path, synthetic_root: Path):
    from pytorch_lightning.callbacks import LearningRateMonitor

    from deepcontrail.segmentation.training import prepare_training

    run = prepare_training(
        _config(tmp_path, synthetic_root), output_dir=tmp_path, accelerator="cpu"
    )

    monitors = [c for c in run.trainer.callbacks if isinstance(c, LearningRateMonitor)]
    assert monitors, "the learning-rate schedule must be logged"
    assert monitors[0].logging_interval == "step", "warmup + cosine changes the LR every step"


def test_prepare_training_saves_a_checkpoint_at_the_end_of_every_epoch(
    tmp_path: Path, synthetic_root: Path
):
    from pytorch_lightning.callbacks import ModelCheckpoint

    from deepcontrail.segmentation.training import prepare_training

    run = prepare_training(
        _config(tmp_path, synthetic_root, checkpoint_every_epoch=True),
        output_dir=tmp_path / "runs",
        run_name="ep",
        accelerator="cpu",
    )

    checkpoints = [c for c in run.trainer.callbacks if isinstance(c, ModelCheckpoint)]
    assert len(checkpoints) == 2, "checkpoint_every_epoch adds a second ModelCheckpoint"
    per_epoch = checkpoints[1]
    assert per_epoch.dirpath.endswith("epochs"), (
        f"the per-epoch weights go in a dedicated epochs/ dir, got {per_epoch.dirpath}"
    )
    assert per_epoch.save_top_k == -1, "every epoch's weights must be kept, for averaging later"
    assert per_epoch.save_weights_only is True, "only weights are saved for averaging"
    assert checkpoints[0].monitor == "val_dice", (
        "the best/last checkpoint stays first and unchanged"
    )


def test_prepare_training_records_the_exact_train_and_val_records(
    tmp_path: Path, synthetic_root: Path
):
    from deepcontrail.core.splits import split_fingerprint
    from deepcontrail.segmentation.training import prepare_training

    run = prepare_training(
        _config(tmp_path, synthetic_root), output_dir=tmp_path, run_name="r", accelerator="cpu"
    )

    train = (run.run_dir / "split" / "train.txt").read_text().split()
    val = (run.run_dir / "split" / "val.txt").read_text().split()
    assert train == run.train_loader.dataset.stems  # type: ignore[attr-defined]
    assert val == run.val_loader.dataset.stems  # type: ignore[attr-defined]
    hparams = run.trainer.logger.experiment.hparams  # type: ignore[union-attr]
    assert hparams["data/val_split_fingerprint"] == split_fingerprint(val), (
        "the logger records which validation records the run used"
    )
    assert hparams["data/train_split_fingerprint"] == split_fingerprint(train)
