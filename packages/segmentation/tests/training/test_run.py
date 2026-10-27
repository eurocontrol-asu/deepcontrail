"""Tests for deepcontrail.segmentation.training.run — the training entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
    from deepcontrail.segmentation.training import build_logger

    uri = f"sqlite:///{tmp_path}/given.db"

    logger = build_logger("mlflow", output_dir=tmp_path, run_name="r1", mlflow_uri=uri)

    assert logger._tracking_uri == uri  # type: ignore[attr-defined]


def test_build_logger_mlflow_falls_back_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from deepcontrail.segmentation.training import build_logger

    uri = f"sqlite:///{tmp_path}/env.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    logger = build_logger("mlflow", output_dir=tmp_path, run_name="r1")

    assert logger._tracking_uri == uri  # type: ignore[attr-defined]


def test_build_logger_mlflow_defaults_to_a_database_in_the_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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
