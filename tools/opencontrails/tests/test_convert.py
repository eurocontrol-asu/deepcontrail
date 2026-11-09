"""Tests for the TFRecord converter's per-frame outputs."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from deepcontrail.core.contract import DatasetInfo


def _bands_dict(n_frames: int = 1) -> dict[str, np.ndarray]:
    from deepcontrail.opencontrails.bands import SPECTRAL_BAND_KEYS

    return {
        key: np.full((2, 2, n_frames), 200.0 + i, dtype=np.float64)
        for i, key in enumerate(SPECTRAL_BAND_KEYS)
    }


def test_spectral_bands_frame_stacks_keys_in_shared_channel_order():
    from deepcontrail.opencontrails.convert import _spectral_bands_frame

    bands = _bands_dict()

    frame = _spectral_bands_frame(bands, 0)

    assert frame.shape == (2, 2, 10)
    assert frame.dtype == np.float32
    np.testing.assert_array_equal(
        frame[0, 0],
        200.0 + np.arange(10),
        err_msg="channel i must be SPECTRAL_BAND_KEYS[i], the order the data loader assumes",
    )


def test_spectral_bands_frame_reads_requested_frame():
    from deepcontrail.opencontrails.convert import _spectral_bands_frame

    bands = _bands_dict(n_frames=2)
    bands["cloud_top"][..., 1] = 999.0

    frame = _spectral_bands_frame(bands, 1)

    assert np.all(frame[..., 4] == 999.0), "frame t must come from time index t"


def _spectral_frame(**values: float) -> np.ndarray:
    from deepcontrail.opencontrails.bands import SPECTRAL_BAND_KEYS

    base = dict.fromkeys(SPECTRAL_BAND_KEYS, 260.0) | values
    return np.stack([np.full((2, 2), base[k], dtype=np.float32) for k in SPECTRAL_BAND_KEYS], -1)


def test_preview_rgb_uses_standard_ash_recipe():
    from deepcontrail.opencontrails.convert import _preview_rgb

    spectral = _spectral_frame(data_12um=269.0, data_11um=270.0, cloud_top=268.0, data_10um=275.0)

    preview = _preview_rgb(spectral)

    assert preview.dtype == np.uint8
    expected = [(269 - 270 + 4) / 6, (270 - 268 + 4) / 9, (270 - 243) / 60]
    for c, value in enumerate(expected):
        assert np.all(preview[..., c] == np.uint8(np.float32(value) * 255)), (
            f"channel {c}: preview must be the standard ash RGB (G = band14 - band11 = cloud_top)"
        )


def test_preview_rgb_when_data_10um_changes_preview_is_unchanged():
    from deepcontrail.opencontrails.convert import _preview_rgb

    low = _preview_rgb(_spectral_frame(data_10um=250.0))
    high = _preview_rgb(_spectral_frame(data_10um=290.0))

    assert np.array_equal(low, high), "data_10um (band 13) is not part of the standard ash recipe"


def test_convert_tfrecords_import_does_not_load_tensorflow():
    code = "import sys, deepcontrail.opencontrails.convert; print('tensorflow' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False", (
        "tensorflow must be imported lazily: loading it next to torch segfaults the test suite"
    )


def test_prepare_output_dir_writes_a_format_v1_band_manifest(tmp_path: Path):
    from deepcontrail.core.contract import read_band_manifest
    from deepcontrail.opencontrails.bands import SPECTRAL_BAND_INFO, SPECTRAL_BAND_KEYS
    from deepcontrail.opencontrails.convert import _prepare_output_dir

    _prepare_output_dir(tmp_path)

    manifest = read_band_manifest(tmp_path)
    assert manifest.sensor == "GOES-16 ABI"
    assert [c.key for c in manifest.channels] == list(SPECTRAL_BAND_KEYS), (
        "channel i = SPECTRAL_BAND_KEYS[i]"
    )
    assert [c.sensor_band for c in manifest.channels] == [
        None if SPECTRAL_BAND_INFO[k].abi_band is None else str(SPECTRAL_BAND_INFO[k].abi_band)
        for k in SPECTRAL_BAND_KEYS
    ], "data_10um is ABI band 13, cloud_top band 11"
    assert {c.units for c in manifest.channels} == {"K"}


def test_prepare_output_dir_creates_train_and_test_directories_only(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _prepare_output_dir

    _prepare_output_dir(tmp_path)

    for sub in ("images_semantic", "semantic_masks", "probability_masks", "spectral_bands"):
        for split in ("train", "test"):
            assert (tmp_path / sub / split).is_dir(), f"{sub}/{split} must exist before conversion"
        assert not (tmp_path / sub / "val").exists(), (
            "validation is drawn from train/ at load time; a val/ folder would never be read"
        )
    assert (tmp_path / "metadata").is_dir()


def test_prepare_output_dir_is_idempotent(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _prepare_output_dir

    _prepare_output_dir(tmp_path)
    first = (tmp_path / "spectral_bands" / "bands.json").read_text()

    _prepare_output_dir(tmp_path)

    assert (tmp_path / "spectral_bands" / "bands.json").read_text() == first, (
        "re-runs must not change it"
    )


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_record_stem_is_the_record_id():
    from deepcontrail.opencontrails.convert import _record_stem

    assert _record_stem(8683037494574687240) == "8683037494574687240"


def test_record_stem_without_record_id_raises():
    from deepcontrail.opencontrails.convert import _record_stem

    with pytest.raises(ValueError, match="record_id"):
        _record_stem(None)


@pytest.mark.parametrize(
    ("t", "expected"),
    [(0, "123_b4"), (3, "123_b1"), (4, "123"), (5, "123_a1"), (7, "123_a3")],
)
def test_frame_stem_names_context_frames_by_distance_to_label(t: int, expected: str):
    from deepcontrail.opencontrails.convert import _frame_stem

    assert _frame_stem("123", t, label_idx=4, n_frames=8) == expected


def test_frame_stem_single_frame_keeps_record_stem():
    from deepcontrail.opencontrails.convert import _frame_stem

    assert _frame_stem("123", 0, label_idx=0, n_frames=1) == "123"


# ---------------------------------------------------------------------------
# Mask conversions
# ---------------------------------------------------------------------------


def _individual() -> np.ndarray:
    m = np.zeros((2, 2, 1, 4), dtype=np.int32)
    m[0, 0, 0, :] = 1  # all 4 annotators
    m[0, 1, 0, :2] = 1  # 2 of 4
    return m


def test_probability_mask_is_mean_of_annotators():
    from deepcontrail.opencontrails.convert import _probability_mask

    prob = _probability_mask(_individual())

    assert prob.dtype == np.float32
    np.testing.assert_array_equal(prob, [[1.0, 0.5], [0.0, 0.0]])


def test_annotator_masks_keep_one_channel_per_annotator():
    from deepcontrail.opencontrails.convert import _annotator_masks

    masks = _annotator_masks(_individual())

    assert masks.shape == (2, 2, 4)
    assert masks.dtype == np.uint8
    np.testing.assert_array_equal(masks[0, 1], [1, 1, 0, 0])


def test_instance_masks_collapse_trailing_dims():
    from deepcontrail.opencontrails.convert import _instance_masks

    inst = np.zeros((3, 2, 2, 1, 5), dtype=np.int32)

    out = _instance_masks(inst)

    assert out.shape == (3, 2, 2, 5)
    assert out.dtype == np.int8


def test_instance_masks_without_instances_is_empty():
    from deepcontrail.opencontrails.convert import _instance_masks

    out = _instance_masks(np.zeros((0, 2, 2, 1, 4), dtype=np.int32))

    assert out.shape == (0, 2, 2, 4)


def test_consensus_mask_png_maps_one_to_255():
    from deepcontrail.opencontrails.convert import _consensus_mask_png

    png = _consensus_mask_png(np.array([[[1], [0]]], dtype=np.int32))

    assert png.dtype == np.uint8
    np.testing.assert_array_equal(png, [[255, 0]])


def test_verify_consensus_accepts_matching_png(tmp_path: Path):
    from PIL import Image

    from deepcontrail.opencontrails.convert import _verify_consensus

    Image.fromarray(np.array([[255, 0]], dtype=np.uint8), mode="L").save(tmp_path / "m.png")

    _verify_consensus(tmp_path / "m.png", np.array([[[1], [0]]]), "123")


def test_verify_consensus_rejects_different_png(tmp_path: Path):
    from PIL import Image

    from deepcontrail.opencontrails.convert import _verify_consensus

    Image.fromarray(np.array([[0, 0]], dtype=np.uint8), mode="L").save(tmp_path / "m.png")

    with pytest.raises(ValueError, match="123"):
        _verify_consensus(tmp_path / "m.png", np.array([[[1], [0]]]), "123")


# ---------------------------------------------------------------------------
# Writing, resume and CLI
# ---------------------------------------------------------------------------


def test_save_npy_writes_file_and_leaves_no_temp(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _save_npy

    _save_npy(tmp_path / "a.npy", np.arange(3))

    np.testing.assert_array_equal(np.load(tmp_path / "a.npy"), np.arange(3))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.npy"], "no temp file may remain"


def test_save_image_writes_requested_format(tmp_path: Path):
    from PIL import Image

    from deepcontrail.opencontrails.convert import _save_image

    _save_image(tmp_path / "m.png", Image.fromarray(np.zeros((2, 2), dtype=np.uint8)), "PNG")

    with Image.open(tmp_path / "m.png") as img:
        assert img.format == "PNG"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.png"]


def _touch_record(root: Path, stem: str, frames: list[str], *, skip: str | None = None) -> None:
    (root / "semantic_masks" / "train").mkdir(parents=True, exist_ok=True)
    (root / "spectral_bands" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images_semantic" / "train").mkdir(parents=True, exist_ok=True)
    (root / "semantic_masks" / "train" / f"{stem}.png").touch()
    for f in frames:
        for sub, ext in (("spectral_bands", ".npy"), ("images_semantic", ".jpg")):
            path = root / sub / "train" / f"{f}{ext}"
            if path.name != skip:
                path.touch()


def test_record_complete_when_mask_and_every_frame_exist(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _record_complete

    frames = ["9_b1", "9", "9_a1"]
    _touch_record(tmp_path, "9", frames)

    assert _record_complete(tmp_path, "train", "9", frames)


def test_record_incomplete_when_a_context_frame_is_missing(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _record_complete

    frames = ["9_b1", "9", "9_a1"]
    _touch_record(tmp_path, "9", frames, skip="9_a1.jpg")

    assert not _record_complete(tmp_path, "train", "9", frames), (
        "an interrupted clip must be redone, not skipped"
    )


def test_parse_args_skips_existing_by_default():
    from deepcontrail.opencontrails.convert import _parse_args

    args = _parse_args(["--tfrecord-dir", "a", "--output-dir", "b"])

    assert args.skip_existing is True


def test_parse_args_no_skip_existing_forces_rewrite():
    from deepcontrail.opencontrails.convert import _parse_args

    args = _parse_args(["--tfrecord-dir", "a", "--output-dir", "b", "--no-skip-existing"])

    assert args.skip_existing is False


def test_list_shards_filters_by_prefix_and_glob(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _list_shards

    for name in (
        "train.tfrecords-00001-of-00002",
        "train.tfrecords-00000-of-00002",
        "validation.tfrecords-00000-of-00001",
    ):
        (tmp_path / name).touch()

    assert [p.name for p in _list_shards(tmp_path, "train.tfrecords", None)] == [
        "train.tfrecords-00000-of-00002",
        "train.tfrecords-00001-of-00002",
    ]
    assert [p.name for p in _list_shards(tmp_path, "train.tfrecords", "*-00001-*")] == [
        "train.tfrecords-00001-of-00002"
    ]


def test_write_split_lists_from_masks_on_disk(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _write_split_lists

    for split, stems in (("train", ["20", "3"]), ("test", ["7"])):
        d = tmp_path / "semantic_masks" / split
        d.mkdir(parents=True)
        for st in stems:
            (d / f"{st}.png").touch()

    _write_split_lists(tmp_path)

    assert (tmp_path / "splits_train.txt").read_text().split() == ["20", "3"]
    assert (tmp_path / "splits_test.txt").read_text().split() == ["7"]


# ---------------------------------------------------------------------------
# dataset_info.json
# ---------------------------------------------------------------------------


def _info(is_temporal: bool, has_instance_masks: bool) -> DatasetInfo:
    from deepcontrail.opencontrails.convert import FormatInfo, _dataset_info

    return _dataset_info(
        FormatInfo(is_temporal=is_temporal, has_instance_masks=has_instance_masks),
        {"train": 3, "test": 1},
        ("semantic", "probability"),
    )


def test_dataset_info_temporal_export_recommends_half_pixel_shift():
    info = _info(is_temporal=True, has_instance_masks=False)

    assert info.export == "20230419"
    assert info.recommended_pixel_shift == 0.5, "2023 masks follow the Kaggle convention"


def test_dataset_info_single_frame_export_recommends_no_shift():
    info = _info(is_temporal=False, has_instance_masks=True)

    assert info.export == "20250305"
    assert info.recommended_pixel_shift == 0.0, "2025 masks are already aligned with the images"


def test_dataset_info_unknown_format_has_no_recommendation():
    info = _info(is_temporal=False, has_instance_masks=False)

    assert info.export is None
    assert info.recommended_pixel_shift is None, "never guess a shift for an unmeasured export"


def test_dataset_info_records_format_counts_labels_and_naming():
    info = _info(is_temporal=True, has_instance_masks=False)

    assert info.format_version == 1
    assert info.temporal is True
    assert info.records == {"train": 3, "test": 1}
    assert info.labels == ("semantic", "probability")
    assert info.naming == "record_id"


def _converted_root(tmp_path: Path, *, temporal: bool = False, instance: bool = True) -> Path:
    """Build a tiny already-converted dataset: one train and one test record."""
    from deepcontrail.opencontrails.convert import _prepare_output_dir

    _prepare_output_dir(tmp_path)
    for split, stem in (("train", "11"), ("test", "22")):
        np.save(
            tmp_path / "spectral_bands" / split / f"{stem}.npy", np.zeros((2, 2, 10), np.float32)
        )
        if temporal:
            np.save(tmp_path / "spectral_bands" / split / f"{stem}_b1.npy", np.zeros((2, 2, 10)))
        (tmp_path / "semantic_masks" / split / f"{stem}.png").touch()
        np.save(tmp_path / "probability_masks" / split / f"{stem}.npy", np.zeros((2, 2)))
        if instance:
            np.save(tmp_path / "instance_masks" / split / f"{stem}.npy", np.zeros((0, 2, 2, 1)))
        (tmp_path / f"splits_{split}.txt").write_text(f"{stem}\n")
    return tmp_path


def test_labels_on_disk_lists_only_label_directories_with_files(tmp_path: Path):
    from deepcontrail.opencontrails.convert import _labels_on_disk

    root = _converted_root(tmp_path, instance=False)

    assert _labels_on_disk(root) == ("semantic", "probability"), (
        "empty label directories (annotator, instance here) are not labels"
    )


@pytest.mark.parametrize(("temporal", "instance"), [(False, True), (True, False)])
def test_format_from_disk_detects_context_frames_and_instance_masks(
    tmp_path: Path, temporal: bool, instance: bool
):
    from deepcontrail.opencontrails.convert import FormatInfo, _format_from_disk

    root = _converted_root(tmp_path, temporal=temporal, instance=instance)

    assert _format_from_disk(root) == FormatInfo(is_temporal=temporal, has_instance_masks=instance)


def test_write_dataset_info_counts_records_and_labels_on_disk(tmp_path: Path):
    from deepcontrail.core.contract import read_dataset_info
    from deepcontrail.opencontrails.convert import FormatInfo, _write_dataset_info

    root = _converted_root(tmp_path)

    _write_dataset_info(root, FormatInfo(is_temporal=False, has_instance_masks=True))

    info = read_dataset_info(root)
    assert info.records == {"train": 1, "test": 1}
    assert info.labels == ("semantic", "probability", "instance")
    assert info.recommended_pixel_shift == 0.0


def test_parse_args_requires_tfrecord_dir_for_a_conversion():
    from deepcontrail.opencontrails.convert import _parse_args

    with pytest.raises(SystemExit):
        _parse_args(["--output-dir", "out"])


def test_parse_args_metadata_only_needs_no_tfrecord_dir():
    from deepcontrail.opencontrails.convert import _parse_args

    args = _parse_args(["--metadata-only", "--output-dir", "out"])

    assert args.metadata_only is True
    assert args.tfrecord_dir is None


def test_metadata_only_rewrites_both_json_files_and_nothing_else(tmp_path: Path):
    import json

    from deepcontrail.core.contract import read_band_manifest, read_dataset_info
    from deepcontrail.opencontrails.convert import main

    root = _converted_root(tmp_path)
    legacy = {"dtype": "float32", "units": "K", "channels": []}
    (root / "spectral_bands" / "bands.json").write_text(json.dumps(legacy))
    (root / "metadata" / "dataset_info.json").write_text(json.dumps({"export": "20250305"}))
    jsons = {"spectral_bands/bands.json", "metadata/dataset_info.json"}
    before = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in jsons
    }

    with mock.patch.dict("sys.modules", {"tensorflow": None}):
        assert main(["--metadata-only", "--output-dir", str(root)]) == 0

    after = {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and str(p.relative_to(root)) not in jsons
    }
    assert after == before, "--metadata-only must not touch anything but the two JSON files"
    assert read_band_manifest(root).format_version == 1
    info = read_dataset_info(root)
    assert info.export == "20250305"
    assert info.records == {"train": 1, "test": 1}


def test_metadata_only_without_split_lists_fails_clearly(tmp_path: Path):
    from deepcontrail.opencontrails.convert import main

    root = _converted_root(tmp_path)
    (root / "splits_test.txt").unlink()

    with pytest.raises(FileNotFoundError, match=r"splits_test\.txt"):
        main(["--metadata-only", "--output-dir", str(root)])


def test_main_with_a_missing_tfrecord_dir_fails_and_creates_nothing(tmp_path: Path):
    from deepcontrail.opencontrails.convert import main

    code = main(
        ["--tfrecord-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out")]
    )

    assert code == 2, "a missing input folder is an error, not a successful empty conversion"
    assert not (tmp_path / "out").exists(), (
        "--output-dir must not be created before the input folder is validated"
    )


def test_main_without_matching_shards_fails_and_creates_nothing(tmp_path: Path):
    from deepcontrail.opencontrails.convert import main

    tf_dir = tmp_path / "tfrecords"
    tf_dir.mkdir()

    code = main(["--tfrecord-dir", str(tf_dir), "--output-dir", str(tmp_path / "out")])

    assert code == 2, "a folder with no matching shards is an error, not an empty conversion"
    assert not (tmp_path / "out").exists(), (
        "--output-dir must not be created before the shards are found"
    )


def test_format_from_disk_keeps_the_declared_temporal_value(tmp_path: Path):
    import json

    from deepcontrail.opencontrails.convert import _format_from_disk

    root = _converted_root(tmp_path)
    # A single-frame record whose id merely looks like a context frame of record 11.
    np.save(tmp_path / "spectral_bands" / "train" / "11_a1.npy", np.zeros((2, 2, 10), np.float32))
    info = {
        "format_version": 1,
        "source": "OpenContrails GOES-16 ABI TFRecords",
        "records": {"train": 2, "test": 1},
        "labels": ["semantic", "probability", "instance"],
        "recommended_pixel_shift": 0.0,
        "temporal": False,
    }
    (root / "metadata" / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")

    fmt = _format_from_disk(root)

    assert fmt.is_temporal is False, "a declared temporal value is kept, not guessed from names"
