"""packages/segmentation/scripts/train.py reads .env, after flags and the real environment.

Each case uses a relative sqlite URI, which the script rejects before doing any work;
the error message shows which of the three sources it took the value from.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "packages" / "segmentation" / "scripts" / "train.py"
VARIABLES = ("DEEPCONTRAIL_CHECKPOINT_ROOT", "MLFLOW_TRACKING_URI")


def _run(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    base = {k: v for k, v in os.environ.items() if k not in VARIABLES}
    result = subprocess.run(
        [sys.executable, str(TRAIN), *args],
        cwd=cwd,
        env=base | (env or {}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "a relative sqlite URI must stop the run"
    return result.stderr


def _write_env(directory: Path, uri: str) -> None:
    (directory / ".env").write_text(
        f"DEEPCONTRAIL_CHECKPOINT_ROOT={directory}/ckpt\nMLFLOW_TRACKING_URI={uri}\n"
    )


def test_train_reads_the_env_file(tmp_path: Path):
    _write_env(tmp_path, "sqlite:///from-file.db")

    stderr = _run(tmp_path)

    assert "sqlite:///from-file.db" in stderr, "the .env value must be used"


def test_real_environment_wins_over_the_env_file(tmp_path: Path):
    _write_env(tmp_path, "sqlite:///from-file.db")

    stderr = _run(tmp_path, env={"MLFLOW_TRACKING_URI": "sqlite:///from-shell.db"})

    assert "sqlite:///from-shell.db" in stderr, "a variable set in the shell must win"


def test_cli_flag_wins_over_everything(tmp_path: Path):
    _write_env(tmp_path, "sqlite:///from-file.db")

    stderr = _run(
        tmp_path,
        "--mlflow-uri",
        "sqlite:///from-flag.db",
        env={"MLFLOW_TRACKING_URI": "sqlite:///from-shell.db"},
    )

    assert "sqlite:///from-flag.db" in stderr, "the CLI flag must win"


def test_env_example_lists_every_variable():
    example = (ROOT / ".env.example").read_text()

    for name in VARIABLES:
        assert f"\n{name}=" in example, f".env.example must document {name}"


def _load_train_script() -> ModuleType:
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_script", TRAIN)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _history(steps_per_epoch: int = 10, epochs: int = 4) -> dict[str, list[tuple[int, float]]]:
    ends = [(e + 1) * steps_per_epoch - 1 for e in range(epochs)]
    return {
        "train_loss": [(s, 1.0 / (i + 1)) for i, s in enumerate(ends)],
        "val_loss": [(s, 1.2 / (i + 1)) for i, s in enumerate(ends)],
        "val_dice": [(s, v) for s, v in zip(ends, [0.5, 0.6, 0.7, 0.65], strict=True)],
        "val_iou": [(s, v) for s, v in zip(ends, [0.3, 0.4, 0.5, 0.45], strict=True)],
        "train_dice": [(s, v) for s, v in zip(ends, [0.55, 0.7, 0.8, 0.85], strict=True)],
        "train_iou": [(s, v) for s, v in zip(ends, [0.35, 0.5, 0.6, 0.7], strict=True)],
        "train_loss_step": [(s, 1.0 / (s + 1)) for s in range(0, epochs * steps_per_epoch, 5)],
        "lr-AdamW": [(s, 1e-4 * s / 40) for s in range(0, epochs * steps_per_epoch, 5)],
    }


def test_training_curves_have_one_panel_per_measure():
    train = _load_train_script()

    fig = train.plot_training_curves(_history(), steps_per_epoch=10, best_epoch=3)

    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert titles == ["Loss", "Dice / IoU", "Learning rate"], (
        "losses, scores and the LR have different scales: one panel each, never a twin axis"
    )


def test_training_curves_plot_epochs_and_mark_the_best_epoch():
    train = _load_train_script()

    fig = train.plot_training_curves(_history(), steps_per_epoch=10, best_epoch=3)

    loss_ax, score_ax, _ = fig.axes
    val_loss = next(line for line in loss_ax.get_lines() if line.get_label() == "val loss")
    assert list(val_loss.get_xdata()) == [1.0, 2.0, 3.0, 4.0], "per-epoch points sit on epochs"
    assert score_ax.get_legend() is not None, "two series need a legend"
    assert any(line.get_label() == "best epoch" for line in score_ax.get_lines())
    labels = {line.get_label() for line in score_ax.get_lines()}
    assert {"train Dice", "val Dice", "train IoU", "val IoU"} <= labels, (
        "train next to val shows overfitting"
    )


def test_training_curves_tolerate_missing_series():
    train = _load_train_script()

    fig = train.plot_training_curves({"val_dice": [(9, 0.5)]}, steps_per_epoch=10, best_epoch=None)

    assert len(fig.axes) == 3, "a run without step losses or LR still gets its figure"
