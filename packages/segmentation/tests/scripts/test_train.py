"""Tests for scripts/train.py — where a run writes its checkpoints and MLflow logs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"

# The script is the repository's experiment workflow: it needs MLflow and python-dotenv, which
# the package does not (they come with the repository's experiment tools).
pytest.importorskip("mlflow")
pytest.importorskip("dotenv")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- checkpoint root and MLflow URI (no user-specific defaults) ---


def test_resolve_run_paths_prefers_explicit_values_over_the_environment(tmp_path) -> None:
    resolve_run_paths = _load_script().resolve_run_paths

    env = {"DEEPCONTRAIL_CHECKPOINT_ROOT": "/env/ckpt", "MLFLOW_TRACKING_URI": "sqlite:////env.db"}

    root, uri = resolve_run_paths(tmp_path, "sqlite:////flag.db", environ=env)

    assert (root, uri) == (tmp_path, "sqlite:////flag.db")


def test_resolve_run_paths_falls_back_to_the_environment() -> None:
    from pathlib import Path

    resolve_run_paths = _load_script().resolve_run_paths

    env = {"DEEPCONTRAIL_CHECKPOINT_ROOT": "/env/ckpt", "MLFLOW_TRACKING_URI": "sqlite:////env.db"}

    root, uri = resolve_run_paths(None, None, environ=env)

    assert (root, uri) == (Path("/env/ckpt"), "sqlite:////env.db")


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        ({"MLFLOW_TRACKING_URI": "sqlite:////a.db"}, "DEEPCONTRAIL_CHECKPOINT_ROOT"),
        ({"DEEPCONTRAIL_CHECKPOINT_ROOT": "/ckpt"}, "MLFLOW_TRACKING_URI"),
    ],
)
def test_resolve_run_paths_without_a_value_names_what_to_set(env, missing) -> None:
    resolve_run_paths = _load_script().resolve_run_paths

    with pytest.raises(ValueError, match=missing):
        resolve_run_paths(None, None, environ=env)


def test_resolve_run_paths_rejects_a_relative_sqlite_path() -> None:
    resolve_run_paths = _load_script().resolve_run_paths

    with pytest.raises(ValueError, match="sqlite:////"):
        resolve_run_paths("/ckpt", "sqlite:///data/mlflow.db", environ={})


@pytest.mark.parametrize("uri", ["sqlite:////data/mlflow.db", "http://localhost:5000"])
def test_resolve_run_paths_accepts_absolute_sqlite_and_server_uris(uri) -> None:
    resolve_run_paths = _load_script().resolve_run_paths

    _root, resolved = resolve_run_paths("/ckpt", uri, environ={})

    assert resolved == uri
