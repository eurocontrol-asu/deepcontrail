"""Cross-package invariants of the uv workspace (versions, licences, namespace layout)."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

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

    assert names == {
        "deepcontrail-core",
        "deepcontrail-segmentation",
        "deepcontrail-instance",
        "deepcontrail-opencontrails",
    }


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
    import deepcontrail.instance
    import deepcontrail.segmentation

    assert deepcontrail.core.__version__ == _project(ROOT / "packages/core")["project"]["version"]
    assert (
        deepcontrail.segmentation.__version__
        == _project(ROOT / "packages/segmentation")["project"]["version"]
    )
    assert (
        deepcontrail.instance.__version__
        == _project(ROOT / "packages/instance")["project"]["version"]
    )


def test_pyright_checks_every_package_source_tree():
    include = set(_project(ROOT)["tool"]["pyright"]["include"])

    expected = {str((m / "src").relative_to(ROOT)) for m in MEMBERS}

    assert include == expected, (
        "pyright must list each package's src/ explicitly: glob patterns matched no files"
    )


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_ships_a_py_typed_marker(member: Path):
    module = _project(member)["tool"]["uv"]["build-backend"]["module-name"]
    package_dir = member / "src" / Path(*module.split("."))

    assert (package_dir / "py.typed").is_file(), (
        "PEP 561: without py.typed, users' type checkers ignore the package's annotations"
    )


def test_tracked_configs_hold_no_machine_specific_paths():
    configs = [
        p
        for p in ROOT.rglob("*.yaml")
        if not any(
            part.startswith(".") or part in {"node_modules"} for part in p.relative_to(ROOT).parts
        )
        and "/tests/" not in str(p)
    ]

    offending = [
        str(p.relative_to(ROOT))
        for p in configs
        if any(marker in p.read_text(encoding="utf-8") for marker in ("/data/", "/home/"))
    ]

    assert configs, "the repository ships example configs"
    assert offending == [], f"configs must not point at one machine's paths: {offending}"


# What PyPI shows and installs. The package READMEs are the PyPI pages, and pip users get the
# newest dependency versions the bounds allow, not the lock.

REQUIRED_URLS = {"Homepage", "Documentation", "Repository", "Issues", "Changelog"}


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_metadata_points_back_to_the_project(member: Path):
    project = _project(member)["project"]

    missing = REQUIRED_URLS - set(project.get("urls", {}))

    assert not missing, f"the PyPI page links to the project; missing URLs: {sorted(missing)}"
    assert project.get("keywords"), "keywords make the package findable on PyPI"
    assert any(
        c.startswith("Programming Language :: Python :: 3.") for c in project.get("classifiers", [])
    ), "classifiers state the supported Python versions"


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_readme_links_work_outside_the_repository(member: Path):
    readme = (member / "README.md").read_text(encoding="utf-8")

    relative = re.findall(r"\]\((?!https?://|mailto:|#)([^)]+)\)", readme)

    assert relative == [], f"relative links break on the PyPI page; use absolute URLs: {relative}"


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_readme_installs_from_pypi(member: Path):
    readme = (member / "README.md").read_text(encoding="utf-8")

    assert "git+https://" not in readme, "the package is installed from PyPI, not from git"
    assert "uv sync" not in readme, "uv sync only works in a clone of the repository"


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_does_not_pin_dependencies_exactly(member: Path):
    dependencies = _project(member)["project"]["dependencies"]

    pinned = [d for d in dependencies if "==" in d]

    assert pinned == [], f"an exact pin blocks pip on other Python versions and platforms: {pinned}"


def test_members_share_one_build_backend_requirement():
    requirements = {m.name: _project(m)["build-system"]["requires"] for m in MEMBERS}

    assert len({tuple(r) for r in requirements.values()}) == 1, (
        f"every package builds with the same backend bounds: {requirements}"
    )


def _python_versions_tested_in_ci() -> dict[str, set[str]]:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    tested: dict[str, set[str]] = {}
    for job in workflow["jobs"]["package"]["strategy"]["matrix"]["include"]:
        tested.setdefault(job["package"], set()).add(job["python"])
    return tested


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_ci_tests_the_oldest_and_newest_python_each_member_claims(member: Path):
    project = _project(member)["project"]
    prefix = "Programming Language :: Python :: "
    claimed = sorted(
        (
            c.removeprefix(prefix)
            for c in project.get("classifiers", [])
            if re.fullmatch(r".*:: 3\.\d+", c)
        ),
        key=lambda v: int(v.split(".")[1]),
    )

    tested = _python_versions_tested_in_ci().get(project["name"], set())

    assert claimed, "classifiers state the supported Python versions"
    assert {claimed[0], claimed[-1]} <= tested, (
        f"{project['name']} claims Python {claimed[0]}-{claimed[-1]}; CI must test both ends "
        f"(tested: {sorted(tested)})"
    )


@pytest.mark.parametrize("job", ["pypi", "testpypi"])
def test_release_publishes_each_package_independently(job: str):
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    strategy = workflow["jobs"][job]["strategy"]

    assert strategy.get("fail-fast") is False, (
        "one package failing to publish (e.g. its trusted publisher is not registered yet) must "
        "not cancel the uploads of the others; the failed job is re-run on its own"
    )
    assert sorted(strategy["matrix"]["package"]) == sorted(m.name for m in MEMBERS), (
        "every package of the workspace is published"
    )


def test_citation_metadata_matches_the_packages():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    package = _project(MEMBERS[0])["project"]

    authors = [f"{a['family-names']}, {a['given-names']}" for a in citation["authors"]]

    assert str(citation["version"]) == package["version"], "CITATION.cff cites the released version"
    assert citation["license"] == zenodo["license"] == package["license"], (
        "CITATION.cff, .zenodo.json and the packages declare the same licence"
    )
    assert [c["name"] for c in zenodo["creators"]] == authors, (
        "Zenodo lists the same authors, in the same order, as CITATION.cff"
    )
    assert [c["affiliation"] for c in zenodo["creators"]] == [
        a["affiliation"] for a in citation["authors"]
    ], "Zenodo and CITATION.cff give the same affiliations"


def test_readme_doi_badge_cites_the_citation_doi():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    badge_dois = re.findall(r"https://doi\.org/(10\.\d+/[^)\s]+)", readme.split("\n## ", 1)[0])

    assert "doi" in citation, "CITATION.cff carries the Zenodo DOI once a release exists"
    assert badge_dois == [citation["doi"]], (
        "the README's DOI badge and CITATION.cff cite the same (all-versions) DOI"
    )


def test_changelog_releases_are_dated():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    releases = re.findall(r"^## \[(\d+\.\d+\.\d+)\](.*)$", changelog, re.M)

    assert releases, "the changelog lists the releases"
    undated = [v for v, rest in releases if not re.fullmatch(r" - \d{4}-\d{2}-\d{2}", rest)]
    assert not undated, f"each release heading carries its date (Keep a Changelog): {undated}"


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.name)
def test_member_authors_match_the_citation(member: Path):
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    project = _project(member)["project"]

    cited = [f"{a['given-names']} {a['family-names']}" for a in citation["authors"]]

    assert [a["name"] for a in project["authors"]] == cited, (
        "PyPI credits the same people, in the same order, as CITATION.cff and Zenodo"
    )
    assert project.get("maintainers") == [
        {"name": citation["authors"][0]["affiliation"], "email": "github-asu@eurocontrol.int"}
    ], "users contact the team (the authors' unit and its address), not a personal address"
