"""packages/segmentation/scripts/train.py reads .env, after flags and the real environment.

Each case uses a relative sqlite URI, which the script rejects before doing any work;
the error message shows which of the three sources it took the value from.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
