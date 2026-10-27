"""Check a dataset root against the DeepContrail dataset format v1.

Two levels:

- **fast** (default): the two metadata files, and file names only — every labelled
  scene has the label files its declared labels need, no label lacks its scene,
  record counts and ``splits_*.txt`` match what is on disk.
- **deep** (``deep=True``): additionally reads every file — band arrays load, are
  float32 with the manifest's channels and finite; masks match their scene's size
  and hold valid values; all scenes share one size.

Run it from the command line with
``deepcontrail-validate <root> [--task segmentation|instance] [--deep]``.
"""

from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from PIL import Image

from deepcontrail.core.contract import (
    BAND_MANIFEST_PATH,
    DATASET_INFO_PATH,
    BandManifest,
    DatasetContractError,
    DatasetInfo,
    Label,
    read_band_manifest,
    read_dataset_info,
)
from deepcontrail.core.splits import is_sequence_stem

if TYPE_CHECKING:
    from collections.abc import Callable

Task = Literal["segmentation", "instance"]
_T = TypeVar("_T")

SPLITS = ("train", "test")
LABEL_FILES: dict[Label, tuple[str, str]] = {
    "semantic": ("semantic_masks", ".png"),
    "probability": ("probability_masks", ".npy"),
    "annotator": ("annotator_masks", ".npy"),
    "instance": ("instance_masks", ".npy"),
}
"""Directory and file extension of each label kind."""

TASK_LABELS: dict[Task, tuple[Label, ...]] = {
    "segmentation": ("semantic",),
    "instance": ("instance",),
}
"""Label kinds a task cannot work without."""


@dataclasses.dataclass(frozen=True)
class Issue:
    """One violation of the dataset format.

    Parameters
    ----------
    path
        File concerned, relative to the dataset root.
    rule
        Short rule name, e.g. ``"missing-label"`` or ``"mask-values"``.
    message
        What is wrong.
    """

    path: str
    rule: str
    message: str


@dataclasses.dataclass(frozen=True)
class ValidationReport:
    """Outcome of ``validate_dataset``: every issue found, in a stable order."""

    root: str | Path
    issues: tuple[Issue, ...]
    scenes: int = 0
    deep: bool = False

    @property
    def ok(self) -> bool:
        """Whether the dataset follows the format (no issue found)."""
        return not self.issues

    def summary(self, limit: int | None = 20) -> str:
        """Render the report; list at most ``limit`` issues (``None`` for all)."""
        mode = "deep" if self.deep else "fast"
        if self.ok:
            return f"{self.root}: no issues ({self.scenes} labelled scenes, {mode} checks)"
        shown = self.issues if limit is None else self.issues[:limit]
        lines = [f"{self.root}: {len(self.issues)} issue(s) ({mode} checks)"]
        lines += [f"  [{i.rule}] {i.path}: {i.message}" for i in shown]
        if len(shown) < len(self.issues):
            lines.append(f"  ... and {len(self.issues) - len(shown)} more")
        return "\n".join(lines)


def validate_dataset(
    root: str | Path, *, task: Task = "segmentation", deep: bool = False, workers: int = 8
) -> ValidationReport:
    """Check ``root`` against the DeepContrail dataset format v1.

    Parameters
    ----------
    root
        Dataset root.
    task
        Task the dataset is meant for; its required labels must be declared.
    deep
        Also read every file and check its contents (slow on large datasets).
    workers
        Threads used to read files in deep mode.

    Returns
    -------
    ValidationReport
        Every issue found; ``report.ok`` when there is none.
    """
    root = Path(root)
    issues: list[Issue] = []

    manifest = _read_metadata(root, BAND_MANIFEST_PATH, read_band_manifest, issues)
    info = _read_metadata(root, DATASET_INFO_PATH, read_dataset_info, issues)
    if manifest is None or info is None:
        return ValidationReport(root, tuple(issues), deep=deep)

    missing = [label for label in TASK_LABELS[task] if label not in info.labels]
    if missing:
        msg = f"task {task!r} needs labels {missing}; dataset declares {list(info.labels)}"
        issues.append(Issue(DATASET_INFO_PATH, "task-labels", msg))

    scenes_by_split = {split: _labelled_scenes(root, split) for split in SPLITS}
    for split, scenes in scenes_by_split.items():
        issues += _check_split_files(root, split, scenes, info)

    if deep:
        issues += _check_contents(root, scenes_by_split, manifest, info, workers)

    n_scenes = sum(len(s) for s in scenes_by_split.values())
    return ValidationReport(root, tuple(issues), scenes=n_scenes, deep=deep)


def _read_metadata(
    root: Path, relative: str, reader: Callable[[Path], _T], issues: list[Issue]
) -> _T | None:
    try:
        return reader(root)
    except DatasetContractError as exc:
        issues.append(Issue(relative, "metadata", str(exc)))
        return None


def _labelled_scenes(root: Path, split: str) -> list[str]:
    """Stems of ``spectral_bands/{split}/*.npy``, context frames excluded."""
    paths = (root / "spectral_bands" / split).glob("*.npy")
    return sorted(p.stem for p in paths if not is_sequence_stem(p.stem))


def _check_split_files(root: Path, split: str, scenes: list[str], info: DatasetInfo) -> list[Issue]:
    issues: list[Issue] = []
    declared = info.records.get(split, 0)  # type: ignore[call-overload]
    if declared != len(scenes):
        msg = f"{split}: declares {declared} records, found {len(scenes)} labelled scenes"
        issues.append(Issue(DATASET_INFO_PATH, "records", msg))

    scene_set = set(scenes)
    for label in info.labels:
        directory, ext = LABEL_FILES[label]
        present = {p.stem for p in (root / directory / split).glob(f"*{ext}")}
        for stem in scenes:
            if stem not in present:
                msg = f"labelled scene {stem} has no {label} label"
                issues.append(Issue(f"{directory}/{split}/{stem}{ext}", "missing-label", msg))
        for stem in sorted(present - scene_set):
            msg = f"no spectral_bands/{split}/{stem}.npy for this {label} label"
            issues.append(Issue(f"{directory}/{split}/{stem}{ext}", "orphan-label", msg))

    split_list = root / f"splits_{split}.txt"
    if split_list.is_file() and sorted(split_list.read_text().split()) != scenes:
        msg = f"does not list the {len(scenes)} labelled scenes of spectral_bands/{split}/"
        issues.append(Issue(split_list.name, "split-list", msg))
    return issues


def _check_contents(
    root: Path,
    scenes_by_split: dict[str, list[str]],
    manifest: BandManifest,
    info: DatasetInfo,
    workers: int,
) -> list[Issue]:
    jobs = [
        (split, path.stem)
        for split in SPLITS
        for path in sorted((root / "spectral_bands" / split).glob("*.npy"))
    ]
    labelled = {(split, s) for split, scenes in scenes_by_split.items() for s in scenes}

    def check(job: tuple[str, str]) -> tuple[list[Issue], tuple[int, ...] | None]:
        split, stem = job
        labels = info.labels if job in labelled else ()
        return _check_scene(root, split, stem, manifest, labels)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(check, jobs))

    issues: list[Issue] = []
    reference: tuple[tuple[int, ...], str] | None = None
    for (split, stem), (scene_issues, hw) in zip(jobs, results, strict=True):
        issues += scene_issues
        if hw is None:
            continue
        path = f"spectral_bands/{split}/{stem}.npy"
        if reference is None:
            reference = (hw, path)
        elif hw != reference[0]:
            msg = f"scene size {hw} differs from {reference[0]} ({reference[1]})"
            issues.append(Issue(path, "size", msg))
    return issues


def _check_scene(
    root: Path, split: str, stem: str, manifest: BandManifest, labels: tuple[Label, ...]
) -> tuple[list[Issue], tuple[int, ...] | None]:
    """Check one scene's band array and label files; return its issues and (H, W)."""
    issues: list[Issue] = []
    band_path = f"spectral_bands/{split}/{stem}.npy"
    bands = _load(root, band_path, issues)
    if bands is None:
        return issues, None
    n = len(manifest.channels)
    if bands.dtype != np.float32 or bands.ndim != 3 or bands.shape[-1] != n:
        msg = f"expected float32 (H, W, {n}), got {bands.dtype} {bands.shape}"
        issues.append(Issue(band_path, "bands", msg))
        return issues, None
    if not np.isfinite(bands).all():
        issues.append(Issue(band_path, "bands", "contains NaN or infinite values"))
    hw = tuple(bands.shape[:2])

    for label in labels:
        directory, ext = LABEL_FILES[label]
        path = f"{directory}/{split}/{stem}{ext}"
        if not (root / path).is_file():
            continue  # reported by the fast checks
        mask = _load(root, path, issues)
        if mask is None:
            continue
        problem = _LABEL_CHECKS[label](mask)
        if problem:
            issues.append(Issue(path, "mask-values", problem))
        mask_hw = tuple(mask.shape[1:3]) if label == "instance" else tuple(mask.shape[:2])
        if mask.ndim >= 2 and mask_hw != hw:
            issues.append(Issue(path, "size", f"size {mask_hw} differs from its bands {hw}"))
    return issues, hw


def _load(root: Path, relative: str, issues: list[Issue]) -> np.ndarray | None:
    try:
        if relative.endswith(".png"):
            with Image.open(root / relative) as image:
                return np.asarray(image)
        return np.load(root / relative)
    except Exception as exc:  # any corrupt or truncated file
        issues.append(Issue(relative, "unreadable", f"{type(exc).__name__}: {exc}"))
        return None


def _values_within(mask: np.ndarray, allowed: set[int]) -> bool:
    return set(np.unique(mask).tolist()) <= allowed


def _semantic_problem(mask: np.ndarray) -> str | None:
    if mask.ndim != 2:
        return f"expected a single-channel (H, W) PNG, got shape {mask.shape}"
    if not _values_within(mask, {0, 1, 255}):
        return f"values must be 0 and 255 (or 1), got {sorted(np.unique(mask).tolist())[:10]}"
    return None


def _probability_problem(mask: np.ndarray) -> str | None:
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.floating):
        return f"expected float (H, W), got {mask.dtype} {mask.shape}"
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        return "values must lie in [0, 1]"
    return None


def _annotator_problem(mask: np.ndarray) -> str | None:
    if mask.ndim != 3:
        return f"expected (H, W, annotators), got shape {mask.shape}"
    if not _values_within(mask, {0, 1}):
        return "values must be 0 or 1"
    return None


def _instance_problem(mask: np.ndarray) -> str | None:
    if mask.ndim != 4:
        return f"expected (instances, H, W, annotators), got shape {mask.shape}"
    if not _values_within(mask, {-1, 0, 1}):
        return "values must be -1, 0 or 1"
    return None


_LABEL_CHECKS: dict[Label, Callable[[np.ndarray], str | None]] = {
    "semantic": _semantic_problem,
    "probability": _probability_problem,
    "annotator": _annotator_problem,
    "instance": _instance_problem,
}


def main(argv: list[str] | None = None) -> int:
    """Validate a dataset root; print the report; return 1 if any issue was found."""
    parser = argparse.ArgumentParser(
        prog="deepcontrail-validate",
        description="Check a dataset root against the DeepContrail dataset format v1.",
    )
    parser.add_argument("root", type=Path, help="Dataset root.")
    parser.add_argument(
        "--task",
        choices=list(TASK_LABELS),
        default="segmentation",
        help="Task the dataset is meant for (default: segmentation).",
    )
    parser.add_argument(
        "--deep", action="store_true", help="Also read every file and check its contents."
    )
    parser.add_argument("--workers", type=int, default=8, help="Reader threads for --deep.")
    args = parser.parse_args(argv)

    report = validate_dataset(args.root, task=args.task, deep=args.deep, workers=args.workers)
    print(report.summary(limit=None))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
