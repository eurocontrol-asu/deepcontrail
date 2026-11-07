"""Tests for deepcontrail.core.validate — dataset format checks and the validation CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

KEYS = ("ir_87", "ir_105", "ir_123")
SIZE = (8, 8)


def _write_metadata(root: Path, records: dict[str, int], labels: list[str]) -> None:
    (root / "spectral_bands").mkdir(parents=True, exist_ok=True)
    (root / "spectral_bands" / "bands.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "sensor": "synthetic",
                "dtype": "float32",
                "channels": [{"index": i, "key": k, "units": "K"} for i, k in enumerate(KEYS)],
            }
        )
    )
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "dataset_info.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source": "synthetic",
                "records": records,
                "labels": labels,
                "recommended_pixel_shift": 0.0,
            }
        )
    )


def _make_root(root: Path) -> Path:
    """Build a valid dataset: 2 train + 1 test scenes, every label kind, one context frame."""
    rng = np.random.RandomState(0)
    stems = {"train": ["11", "12"], "test": ["21"]}
    for split, ids in stems.items():
        for d in (
            "spectral_bands",
            "semantic_masks",
            "probability_masks",
            "annotator_masks",
            "instance_masks",
        ):
            (root / d / split).mkdir(parents=True, exist_ok=True)
        for stem in ids:
            bands = rng.uniform(230, 300, (*SIZE, len(KEYS))).astype(np.float32)
            np.save(root / "spectral_bands" / split / f"{stem}.npy", bands)
            mask = rng.choice([0, 255], size=SIZE).astype(np.uint8)
            Image.fromarray(mask, mode="L").save(root / "semantic_masks" / split / f"{stem}.png")
            prob = (mask / 255.0).astype(np.float32)
            np.save(root / "probability_masks" / split / f"{stem}.npy", prob)
            annot = np.stack([mask > 0, mask > 0], axis=-1).astype(np.uint8)
            np.save(root / "annotator_masks" / split / f"{stem}.npy", annot)
            inst = np.full((1, *SIZE, 2), -1, dtype=np.int8)
            inst[0, ..., 0] = (mask > 0).astype(np.int8)
            np.save(root / "instance_masks" / split / f"{stem}.npy", inst)
        (root / f"splits_{split}.txt").write_text("".join(f"{s}\n" for s in ids))
    # A context frame: spectral bands only, no labels.
    np.save(root / "spectral_bands" / "train" / "11_b1.npy", np.zeros((*SIZE, 3), np.float32))
    _write_metadata(
        root, {"train": 2, "test": 1}, ["semantic", "probability", "annotator", "instance"]
    )
    return root


def _rules(report: object) -> set[str]:
    return {issue.rule for issue in report.issues}  # type: ignore[attr-defined]


def _validate(root: Path, **kwargs: object) -> object:
    from deepcontrail.core.validate import validate_dataset

    return validate_dataset(root, **kwargs)  # type: ignore[arg-type]


# --- a valid dataset -----------------------------------------------------------------


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("task", ["segmentation", "instance"])
def test_valid_dataset_has_no_issues(tmp_path: Path, deep: bool, task: str):
    root = _make_root(tmp_path)

    report = _validate(root, task=task, deep=deep)

    assert report.ok, f"a valid dataset must pass: {report.issues}"  # type: ignore[attr-defined]


# --- fast checks: metadata, files, counts ----------------------------------------------------


def test_missing_band_manifest_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "spectral_bands" / "bands.json").unlink()

    report = _validate(root)

    assert "metadata" in _rules(report)


def test_invalid_dataset_info_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "metadata" / "dataset_info.json").write_text(json.dumps({"format_version": 1}))

    report = _validate(root)

    assert "metadata" in _rules(report)


def test_labelled_scene_without_its_mask_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "semantic_masks" / "train" / "12.png").unlink()

    report = _validate(root)

    issue = next(i for i in report.issues if i.rule == "missing-label")  # type: ignore[attr-defined]
    assert issue.path == "semantic_masks/train/12.png"


def test_label_without_a_scene_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "semantic_masks" / "train" / "99.png").write_bytes(
        (root / "semantic_masks" / "train" / "11.png").read_bytes()
    )

    report = _validate(root)

    assert "orphan-label" in _rules(report)


def test_context_frames_need_no_labels(tmp_path: Path):
    root = _make_root(tmp_path)

    report = _validate(root)

    assert not any("11_b1" in i.path for i in report.issues)  # type: ignore[attr-defined]


def test_record_count_mismatch_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    _write_metadata(
        root, {"train": 5, "test": 1}, ["semantic", "probability", "annotator", "instance"]
    )

    report = _validate(root)

    assert "records" in _rules(report)


def test_split_list_mismatch_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "splits_train.txt").write_text("11\n")

    report = _validate(root)

    assert "split-list" in _rules(report)


def test_missing_split_lists_are_fine(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "splits_train.txt").unlink()
    (root / "splits_test.txt").unlink()

    report = _validate(root)

    assert report.ok, "splits_*.txt are optional"  # type: ignore[attr-defined]


def test_task_needs_its_label_declared(tmp_path: Path):
    root = _make_root(tmp_path)
    _write_metadata(root, {"train": 2, "test": 1}, ["semantic", "probability", "annotator"])

    report = _validate(root, task="instance")

    assert "task-labels" in _rules(report)


def test_declared_label_must_exist_on_disk(tmp_path: Path):
    root = _make_root(tmp_path)
    for split in ("train", "test"):
        for f in (root / "annotator_masks" / split).iterdir():
            f.unlink()

    report = _validate(root)

    assert "missing-label" in _rules(report), "a declared label kind must cover every scene"


def test_fast_mode_does_not_read_arrays(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "spectral_bands" / "train" / "11.npy").write_bytes(b"not an array")

    report = _validate(root, deep=False)

    assert report.ok, "fast checks only use metadata and file names"  # type: ignore[attr-defined]


# --- deep checks: file contents -------------------------------------------------------


def test_unreadable_array_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    (root / "spectral_bands" / "train" / "11.npy").write_bytes(b"not an array")

    report = _validate(root, deep=True)

    assert "unreadable" in _rules(report)


def test_wrong_channel_count_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    np.save(root / "spectral_bands" / "train" / "11.npy", np.zeros((*SIZE, 2), np.float32))

    report = _validate(root, deep=True)

    assert "bands" in _rules(report)


def test_wrong_band_dtype_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    np.save(root / "spectral_bands" / "train" / "11.npy", np.zeros((*SIZE, 3), np.float64))

    report = _validate(root, deep=True)

    assert "bands" in _rules(report)


def test_non_finite_bands_are_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    bands = np.full((*SIZE, 3), 250.0, np.float32)
    bands[0, 0, 0] = np.nan
    np.save(root / "spectral_bands" / "test" / "21.npy", bands)

    report = _validate(root, deep=True)

    assert "bands" in _rules(report)


def test_scene_size_mismatch_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    np.save(root / "probability_masks" / "train" / "12.npy", np.zeros((4, 4), np.float32))

    report = _validate(root, deep=True)

    assert "size" in _rules(report)


def test_scenes_of_different_sizes_are_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    np.save(root / "spectral_bands" / "test" / "21.npy", np.zeros((4, 4, 3), np.float32))

    report = _validate(root, deep=True)

    assert "size" in _rules(report), "the library assumes one native size per dataset"


def test_png_mask_with_intermediate_values_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    mask = np.full(SIZE, 128, dtype=np.uint8)
    Image.fromarray(mask, mode="L").save(root / "semantic_masks" / "train" / "11.png")

    report = _validate(root, deep=True)

    assert "mask-values" in _rules(report), "128 would silently become contrail when binarized"


def test_png_mask_with_zero_and_one_is_accepted(tmp_path: Path):
    root = _make_root(tmp_path)
    mask = np.zeros(SIZE, dtype=np.uint8)
    mask[0, 0] = 1
    Image.fromarray(mask, mode="L").save(root / "semantic_masks" / "train" / "11.png")

    report = _validate(root, deep=True)

    assert report.ok, f"{{0, 1}} masks are valid too: {report.issues}"  # type: ignore[attr-defined]


def test_probability_mask_out_of_range_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    np.save(root / "probability_masks" / "train" / "11.npy", np.full(SIZE, 1.5, np.float32))

    report = _validate(root, deep=True)

    assert "mask-values" in _rules(report)


def test_instance_mask_with_invalid_values_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    np.save(root / "instance_masks" / "train" / "11.npy", np.full((1, *SIZE, 2), 2, np.int8))

    report = _validate(root, deep=True, task="instance")

    assert "mask-values" in _rules(report)


def test_annotator_mask_with_wrong_shape_is_reported(tmp_path: Path):
    root = _make_root(tmp_path)
    np.save(root / "annotator_masks" / "train" / "11.npy", np.zeros(SIZE, np.uint8))

    report = _validate(root, deep=True)

    assert "mask-values" in _rules(report), "annotator masks are (H, W, A)"


# --- report and CLI ------------------------------------------------------------------------


def test_report_summary_caps_the_listed_issues():
    from deepcontrail.core.validate import Issue, ValidationReport

    report = ValidationReport(
        root="/x", issues=tuple(Issue(f"f{i}", "rule", "msg") for i in range(25))
    )

    summary = report.summary(limit=20)

    assert "f19" in summary
    assert "f20" not in summary
    assert "and 5 more" in summary


def test_cli_exits_zero_on_a_valid_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from deepcontrail.core.validate import main

    root = _make_root(tmp_path)

    code = main([str(root), "--deep"])

    assert code == 0
    assert "no issues" in capsys.readouterr().out


def test_cli_exits_non_zero_and_lists_issues(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from deepcontrail.core.validate import main

    root = _make_root(tmp_path)
    (root / "semantic_masks" / "train" / "12.png").unlink()

    code = main([str(root)])

    assert code == 1
    assert "semantic_masks/train/12.png" in capsys.readouterr().out


def test_validate_dataset_reports_an_id_present_in_both_splits(tmp_path: Path):
    root = _make_root(tmp_path)
    # Copy train scene "11" to test with the same id
    (root / "spectral_bands" / "test" / "11.npy").write_bytes(
        (root / "spectral_bands" / "train" / "11.npy").read_bytes()
    )
    for label_dir in ("semantic_masks", "probability_masks", "annotator_masks", "instance_masks"):
        ext = ".png" if label_dir == "semantic_masks" else ".npy"
        src = root / label_dir / "train" / f"11{ext}"
        if src.exists():
            (root / label_dir / "test" / f"11{ext}").write_bytes(src.read_bytes())
    _write_metadata(
        root, {"train": 2, "test": 2}, ["semantic", "probability", "annotator", "instance"]
    )

    report = _validate(root)

    assert "duplicate-id" in _rules(report), (
        "id present in both train/ and test/ is train/test leakage"
    )
    issues = [i for i in report.issues if i.rule == "duplicate-id"]  # type: ignore[attr-defined]
    assert len(issues) == 1
    assert "11" in issues[0].message


def test_validate_dataset_reports_a_labelled_id_with_the_reserved_suffix(tmp_path: Path):
    root = _make_root(tmp_path)
    # Create a labelled scene with reserved suffix _a1
    rng = np.random.RandomState(0)
    stem = "scene_a1"
    split = "train"
    bands = rng.uniform(230, 300, (*SIZE, len(KEYS))).astype(np.float32)
    np.save(root / "spectral_bands" / split / f"{stem}.npy", bands)
    mask = rng.choice([0, 255], size=SIZE).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(root / "semantic_masks" / split / f"{stem}.png")
    _write_metadata(root, {"train": 3, "test": 1}, ["semantic"])

    report = _validate(root)

    assert "reserved-suffix" in _rules(report), (
        "ids ending in _a<k>/_b<k> are reserved for context frames"
    )
    # Should NOT report orphan-label for this file
    orphan_issues = [i for i in report.issues if i.rule == "orphan-label" and stem in i.path]  # type: ignore[attr-defined]
    assert len(orphan_issues) == 0, "should report reserved-suffix, not orphan-label"
    reserved_issues = [i for i in report.issues if i.rule == "reserved-suffix"]  # type: ignore[attr-defined]
    assert len(reserved_issues) >= 1
    assert (
        stem in reserved_issues[0].message
        or "_a" in reserved_issues[0].message
        or "_b" in reserved_issues[0].message
    )


def test_validate_dataset_reports_metadata_that_is_not_utf8(tmp_path: Path):
    root = _make_root(tmp_path)
    # Write a dataset_info.json with non-UTF-8 bytes (Latin-1 encoded café)
    (root / "metadata" / "dataset_info.json").write_bytes(
        b'{"format_version": 1, "source": "caf\xe9", "records": {"train": 2, "test": 1}, "labels": ["semantic"], "recommended_pixel_shift": 0.0}'
    )

    report = _validate(root)

    assert "metadata" in _rules(report), "non-UTF-8 metadata should be reported as a metadata issue"
    issue = next(i for i in report.issues if i.rule == "metadata")  # type: ignore[attr-defined]
    assert (
        "UTF-8" in issue.message
        or "utf-8" in issue.message
        or "UnicodeDecodeError" in issue.message
    )
