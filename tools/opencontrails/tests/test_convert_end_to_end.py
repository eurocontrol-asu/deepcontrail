"""End-to-end conversion of synthetic TFRecords into a validated dataset.

TensorFlow is imported here (lazily, via ``importorskip``) rather than in ``convert.py``:
these tests build real shards and run ``convert.main`` on them, then validate the result
with ``deepcontrail.core.validate_dataset``. They are skipped where TensorFlow is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

tf = pytest.importorskip("tensorflow")

from deepcontrail.core import read_dataset_info, validate_dataset  # noqa: E402
from deepcontrail.opencontrails import convert  # noqa: E402
from deepcontrail.opencontrails.bands import SPECTRAL_BAND_KEYS  # noqa: E402

H = W = 4


def _bytes(t) -> tf.train.Feature:
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(t).numpy()])
    )


def _int(v) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def _example(record_id, *, frames, n_instances=None, n_before=None, n_after=None, seed=0) -> bytes:
    rng = np.random.RandomState(seed)
    feats = {"record_id": _int(record_id)}
    for i, key in enumerate(SPECTRAL_BAND_KEYS):
        feats[key] = _bytes(tf.constant(200.0 + i + rng.rand(H, W, frames), tf.float64))
    individual = rng.randint(0, 2, (H, W, 1, 3)).astype(np.int32)
    feats["human_individual_masks"] = _bytes(tf.constant(individual))
    pixel = (individual.mean(axis=-1) > 0.5).astype(np.int32)
    feats["human_pixel_masks"] = _bytes(tf.constant(pixel))
    if n_instances is not None:
        inst = np.full((n_instances, H, W, 1, 3), -1, np.int32)
        if n_instances:
            inst[0, ..., 0, 0] = individual[..., 0, 0]
        feats["human_individual_instance_masks"] = _bytes(tf.constant(inst))
    if n_before is not None:
        feats["n_times_before"] = _int(n_before)
        feats["n_times_after"] = _int(n_after)
    return tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString()


def _write_shards(tf_dir, train, test) -> None:
    tf_dir.mkdir(parents=True)
    for name, records in (
        ("train.tfrecords-00000-of-00001", train),
        ("validation.tfrecords-00000-of-00001", test),
    ):
        with tf.io.TFRecordWriter(str(tf_dir / name)) as writer:
            for record in records:
                writer.write(record)


_PER_RECORD_SUBDIRS = (
    "spectral_bands",
    "images_semantic",
    "semantic_masks",
    "probability_masks",
    "annotator_masks",
    "instance_masks",
)


def _data_files(root) -> list[Path]:
    """Per-record output files (``<subdir>/<train|test>/...``): what ``--skip-existing`` protects.

    Excludes ``spectral_bands/bands.json``, ``metadata/dataset_info.json`` and ``splits_*.txt``:
    the converter regenerates those on every run (they describe the dataset as a whole, not
    one record), so a rerun rewrites them even when every record is skipped.
    """
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.parent.name in ("train", "test")
        and p.parent.parent.name in _PER_RECORD_SUBDIRS
    )


def test_convert_single_frame_export_produces_a_valid_dataset(tmp_path):
    tf_dir = tmp_path / "tfrecords"
    out = tmp_path / "out"
    train = [
        _example(101, frames=1, n_instances=1, seed=1),
        _example(102, frames=1, n_instances=0, seed=2),
    ]
    test = [_example(201, frames=1, n_instances=1, seed=3)]
    _write_shards(tf_dir, train, test)

    code = convert.main(["--tfrecord-dir", str(tf_dir), "--output-dir", str(out)])
    report = validate_dataset(out, deep=True)

    assert code == 0, "a full single-frame conversion must succeed"
    assert report.ok, report.summary(limit=None)
    info = read_dataset_info(out)
    assert info.temporal is False, (
        "single-frame records (bands (H, W, 1)) declare no context frames"
    )
    assert info.records == {"train": 2, "test": 1}, "two train records, one test record"
    assert (out / "spectral_bands" / "train" / "101.npy").is_file(), "the labelled scene's bands"
    assert (out / "semantic_masks" / "test" / "201.png").is_file(), (
        "the test scene's consensus mask"
    )


def test_convert_temporal_export_names_context_frames_and_declares_them(tmp_path):
    tf_dir = tmp_path / "tfrecords"
    out = tmp_path / "out"
    train = [_example(301, frames=8, n_before=4, n_after=3, n_instances=1, seed=1)]
    test = [_example(401, frames=8, n_before=4, n_after=3, n_instances=1, seed=2)]
    _write_shards(tf_dir, train, test)

    code = convert.main(["--tfrecord-dir", str(tf_dir), "--output-dir", str(out)])
    report = validate_dataset(out, deep=True)

    assert code == 0, "a temporal conversion must succeed"
    assert report.ok, report.summary(limit=None)
    info = read_dataset_info(out)
    assert info.temporal is True, "an 8-frame clip declares its context frames"
    assert info.records == {"train": 1, "test": 1}
    train_bands = sorted(p.name for p in (out / "spectral_bands" / "train").glob("*.npy"))
    assert set(train_bands) == {
        "301.npy",
        "301_b1.npy",
        "301_b2.npy",
        "301_b3.npy",
        "301_b4.npy",
        "301_a1.npy",
        "301_a2.npy",
        "301_a3.npy",
    }, "the labelled frame keeps the stem; each context frame is named by its distance to it"
    assert len(train_bands) == 8, "every frame of the clip is written"


def test_convert_rerun_skips_complete_records(tmp_path):
    tf_dir = tmp_path / "tfrecords"
    out = tmp_path / "out"
    train = [
        _example(101, frames=1, n_instances=1, seed=1),
        _example(102, frames=1, n_instances=0, seed=2),
    ]
    test = [_example(201, frames=1, n_instances=1, seed=3)]
    _write_shards(tf_dir, train, test)

    assert convert.main(["--tfrecord-dir", str(tf_dir), "--output-dir", str(out)]) == 0
    before = {p.relative_to(out): p.stat().st_mtime_ns for p in _data_files(out)}

    assert convert.main(["--tfrecord-dir", str(tf_dir), "--output-dir", str(out)]) == 0
    after = {p.relative_to(out): p.stat().st_mtime_ns for p in _data_files(out)}

    assert before == after, "a rerun must skip complete records and rewrite no per-record file"
