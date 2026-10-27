"""Cross-package invariants of the uv workspace (versions, licences, namespace layout)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = sorted(
    p.parent
    for p in [*ROOT.glob("packages/*/pyproject.toml"), *ROOT.glob("tools/*/pyproject.toml")]
)


def _project(path: Path) -> dict:
    with (path / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_workspace_has_the_expected_members():
    names = {_project(m)["project"]["name"] for m in MEMBERS}

    assert names == {"deepcontrail-core", "deepcontrail-segmentation", "deepcontrail-opencontrails"}


def test_all_packages_share_one_version():
    versions = {_project(m)["project"]["name"]: _project(m)["project"]["version"] for m in MEMBERS}
    root_version = _project(ROOT)["project"]["version"]

    assert set(versions.values()) == {root_version}, (
        f"packages are versioned in lockstep with the workspace ({root_version}): {versions}"
    )


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_licence_is_a_copy_of_the_root_licence(member: Path):
    assert (member / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes(), (
        "each package ships the repository LICENSE (PEP 639 forbids ../ in license-files)"
    )


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_ships_one_namespace_module(member: Path):
    module = _project(member)["tool"]["uv"]["build-backend"]["module-name"]
    namespace, _, name = module.partition(".")

    assert namespace == "deepcontrail", "every package lives under the deepcontrail namespace"
    assert (member / "src" / namespace / name / "__init__.py").is_file()
    assert not (member / "src" / namespace / "__init__.py").exists(), (
        "deepcontrail/__init__.py would shadow the other packages of the namespace"
    )


def test_installed_versions_match_pyproject():
    import deepcontrail.core
    import deepcontrail.segmentation

    assert deepcontrail.core.__version__ == _project(ROOT / "packages/core")["project"]["version"]
    assert (
        deepcontrail.segmentation.__version__
        == _project(ROOT / "packages/segmentation")["project"]["version"]
    )


def test_pyright_checks_every_package_source_tree():
    include = set(_project(ROOT)["tool"]["pyright"]["include"])

    expected = {str((m / "src").relative_to(ROOT)) for m in MEMBERS}

    assert include == expected, (
        "pyright must list each package's src/ explicitly: glob patterns matched no files"
    )
