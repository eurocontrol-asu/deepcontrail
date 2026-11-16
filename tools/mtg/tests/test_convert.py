"""End-to-end tests for deepcontrail.mtg.convert — labels + images to a valid dataset."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from deepcontrail.mtg.convert import convert
from deepcontrail.mtg.labels import read_sequence
from deepcontrail.mtg.split import day_groups, make_split

if TYPE_CHECKING:
    from pathlib import Path

_SIZE = 8


def _image() -> np.ndarray:
    """Return an ``(8, 8, 3)`` float32 image of plausible Kelvin values with one NaN pixel.

    Values vary across the frame so that a block mean that ignores the NaN differs from a
    naive one; pixel ``(0, 0)`` channel 0 is NaN (no-data).
    """
    rows, cols = np.mgrid[0:_SIZE, 0:_SIZE]
    base = (300.0 + 0.1 * (rows * _SIZE + cols)).astype(np.float32)
    image = np.stack([base, base * 0.9, base * 1.1], axis=-1)
    image[0, 0, 0] = np.nan
    return image


def _square() -> list[list[float]]:
    return [[0.25, 0.25], [0.4, 0.25], [0.4, 0.4], [0.25, 0.4]]


def _frame(
    title: str, npy: str | None, *contrails: tuple[str, str]
) -> tuple[str, str | None, list[tuple[str, str]]]:
    """Build a frame spec: ``(image_title, npy|None, [(key, confidence), ...])``."""
    return title, npy, list(contrails)


def _sequence_json(project: str, zone: int, start: str, end: str, frames: list) -> dict:
    """Build one Encord-style export from frame specs (see :func:`_frame`)."""
    images = []
    key_conf: dict[str, str] = {}
    for title, npy, contrails in frames:
        images.append(
            {
                "image_title": title,
                "image_npy": npy,
                "width": _SIZE,
                "height": _SIZE,
                "contrails": {key: [_square()] for key, _ in contrails},
            }
        )
        for key, confidence in contrails:
            key_conf[key] = confidence
    return {
        "sequence": f"MTG_{start}_{end}_zone{zone}_frame4",
        "project_name": project,
        "images": images,
        "contrail_groups": {key: [f"t_{key}"] for key in key_conf},
        "tracks": {
            f"t_{key}": {"static": {"Confidence Level": confidence}}
            for key, confidence in key_conf.items()
        },
    }


_SPECIES = (
    (
        "MTG_S1",
        30,
        [
            _frame(
                "MTG_ash_202506010800_zone30_frame4_f0",
                "S1_d1_f0.npy",
                ("c0", "High"),
                ("c1", "Low"),
            ),
            _frame("MTG_ash_202506011200_zone30_frame4_f1", "S1_d1_f1.npy", ("c0", "High")),
            _frame("MTG_ash_202506020800_zone30_frame4_f0", "S1_d2_f0.npy", ("c0", "High")),
            _frame("MTG_ash_202506021200_zone30_frame4_f1", None, ("c0", "High")),  # no image
            _frame("MTG_ash_202506030800_zone30_frame4_f0", "S1_d3_f0.npy"),  # no contrails
            _frame("MTG_ash_202506031200_zone30_frame4_f1", "S1_d3_f1.npy", ("c0", "High")),
        ],
    ),
    (
        "MTG_S2",
        40,
        [
            _frame("MTG_ash_202506040800_zone40_frame4_f0", "S2_d4_f0.npy", ("c0", "High")),
            _frame("MTG_ash_202506041200_zone40_frame4_f1", "S2_d4_f1.npy", ("c0", "Low")),
            _frame("MTG_ash_202506050800_zone40_frame4_f0", "S2_d5_f0.npy", ("c0", "High")),
            _frame("MTG_ash_202506051200_zone40_frame4_f1", "S2_d5_f1.npy", ("c0", "High")),
            _frame("MTG_ash_202506060800_zone40_frame4_f0", "S2_d6_f0.npy", ("c0", "High")),
            _frame("MTG_ash_202506061200_zone40_frame4_f1", "S2_d6_f1.npy", ("c0", "High")),
        ],
    ),
)


def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Write two projects x three days of tiny sequences and their images; return the dirs."""
    labels_dir = tmp_path / "labels"
    images_dir = tmp_path / "images" / "frames"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    for project, zone, frames in _SPECIES:
        start = frames[0][0].split("_")[2]
        (labels_dir / f"{project}.json").write_text(
            json.dumps(_sequence_json(project, zone, start, "202506011600", frames)),
            encoding="utf-8",
        )
        for _title, npy, _contrails in frames:
            if npy is not None:
                np.save(images_dir / npy, _image())

    return labels_dir, images_dir


def _replay_split(labels_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Recompute, from the exports, each scene's group and on-disk split (as convert() does)."""
    sequences = [read_sequence(path) for path in sorted(labels_dir.glob("*.json"))]
    groups = day_groups(sequences)
    assignment = make_split(sequences, groups, seed=0)
    scene_split: dict[str, str] = {}
    scene_group: dict[str, str] = {}
    for seq in sequences:
        group_id = groups[seq.name]
        split = assignment[group_id]
        disk = "train" if split == "val" else split
        for frame in seq.frames:
            if frame.npy is None:
                continue
            scene_split[frame.title] = disk
            scene_group[frame.title] = group_id
    return scene_split, scene_group


def _label_confidences(labels_dir: Path) -> dict[str, dict[str, str]]:
    """Return, per frame title, ``{contrail key: confidence}`` straight from the raw exports."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(labels_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        groups = data.get("contrail_groups") or {}
        tracks = data.get("tracks") or {}
        level_of = {
            tid: tracks[tid]["static"].get("Confidence Level")
            for tid in tracks
            if isinstance(tracks.get(tid), dict)
        }
        for image in data.get("images", []):
            title = image["image_title"].rsplit(".", 1)[0]
            out[title] = {
                key: level_of.get(track_id)
                for key in (image.get("contrails") or {})
                for track_id in groups.get(key, [])
            }
    return out


def _stems(directory: Path) -> list[str]:
    return sorted(p.name[: -len(".npy")] for p in directory.glob("*.npy") if p.is_file())


def _read_list(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split()


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    return _build_dataset(tmp_path)


@pytest.fixture
def converted(dataset: tuple[Path, Path], tmp_path: Path) -> Path:
    out = tmp_path / "out"
    convert(dataset[0], dataset[1], out, size=_SIZE, workers=1, seed=0)
    return out


def test_convert_writes_a_valid_dataset(converted: Path):
    from deepcontrail.core.validate import validate_dataset

    report = validate_dataset(converted, task="instance", deep=True)
    assert report.ok, report.summary(limit=None)


def test_convert_places_every_scene_in_the_expected_split(
    converted: Path, dataset: tuple[Path, Path]
):
    scene_split, _scene_group = _replay_split(dataset[0])

    train_ids = set(_stems(converted / "spectral_bands" / "train"))
    test_ids = set(_stems(converted / "spectral_bands" / "test"))

    assert train_ids == {id_ for id_, split in scene_split.items() if split == "train"}
    assert test_ids == {id_ for id_, split in scene_split.items() if split == "test"}
    assert train_ids.isdisjoint(test_ids)


def test_convert_split_lists_are_consistent(converted: Path, dataset: tuple[Path, Path]):
    scene_split, _scene_group = _replay_split(dataset[0])
    train_dir = set(_stems(converted / "spectral_bands" / "train"))

    val_ids = _read_list(converted / "splits_val.txt")
    test_ids = _read_list(converted / "splits_test.txt")
    train_ids = _read_list(converted / "splits_train.txt")

    assert set(test_ids) == {id_ for id_, s in scene_split.items() if s == "test"}
    assert set(train_ids) == train_dir, (
        "splits_train.txt lists every id under train/ (val included)"
    )
    assert set(val_ids) <= train_dir, "val scenes live under train/ on disk"


def test_convert_never_splits_a_day_group(converted: Path):
    split = json.loads((converted / "metadata" / "split.json").read_text())
    scenes = json.loads((converted / "metadata" / "scenes.json").read_text())
    group_of = {id_: entry["group"] for id_, entry in scenes.items()}

    splits_by_group: dict[str, set] = {}
    for id_, entry in scenes.items():
        splits_by_group.setdefault(group_of[id_], set()).add(entry["split"])

    assert all(len(splits) == 1 for splits in splits_by_group.values()), (
        "every day-group is entirely in one split"
    )
    expected_groups = {entry["group"]: entry["split"] for entry in scenes.values()}
    assert split["groups"] == expected_groups, "split.json records each group's split"


def test_convert_instance_and_semantic_masks_match(converted: Path):
    from PIL import Image

    for split in ("train", "test"):
        for stem in _stems(converted / "spectral_bands" / split):
            instance = np.load(converted / "instance_masks" / split / f"{stem}.npy")
            assert instance.ndim == 4
            assert instance.shape[1:3] == (_SIZE, _SIZE)
            assert instance.dtype == np.int8
            assert set(np.unique(instance).tolist()) <= {-1, 0, 1}

            union = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
            for n in range(instance.shape[0]):
                union[instance[n, ..., 0] == 1] = 255
            png = np.array(
                Image.open(converted / "semantic_masks" / split / f"{stem}.png").convert("L")
            )
            np.testing.assert_array_equal(png, union, err_msg=f"{split}/{stem}: PNG is the union")


def test_convert_scenes_json_records_label_confidences(converted: Path, dataset: tuple[Path, Path]):
    scenes = json.loads((converted / "metadata" / "scenes.json").read_text())
    labels = _label_confidences(dataset[0])

    for stem, entry in scenes.items():
        kept = {c["key"]: c["confidence"] for c in entry["contrails"]}
        assert set(kept) <= set(labels[stem]), f"{stem}: only labelled contrails are kept"
        for key, confidence in kept.items():
            assert confidence == labels[stem][key], f"{stem}/{key}: confidence matches the label"


def test_convert_records_scene_provenance(converted: Path):
    scenes = json.loads((converted / "metadata" / "scenes.json").read_text())

    day1 = scenes["MTG_ash_202506010800_zone30_frame4_f0"]
    assert day1["project"] == "MTG_S1"
    assert day1["sequence"].startswith("MTG_202506010800")
    assert day1["utm_zone"] == 30
    assert day1["frame_band"] == 4
    assert day1["time"] == "2025-06-01T08:00:00", "the _YYYYmmddHHMM_ token of the title"
    assert day1["labelled_size"] == _SIZE
    assert 0.0 <= day1["nan_fraction"] <= 1.0


def test_convert_is_resumable(dataset: tuple[Path, Path], tmp_path: Path):
    out = tmp_path / "out"
    first = convert(dataset[0], dataset[1], out, size=_SIZE, workers=1, seed=0)
    second = convert(dataset[0], dataset[1], out, size=_SIZE, workers=1, seed=0)

    assert first["frames_written"] > 0
    assert second["frames_written"] == 0, "a second run skips every already-written frame"
    assert second["frames_skipped"] == first["frames_written"]


def test_convert_block_means_down_to_a_smaller_size(tmp_path: Path, dataset: tuple[Path, Path]):
    from deepcontrail.mtg.bands import bands_from_composite
    from deepcontrail.mtg.raster import block_mean

    out = tmp_path / "out4"
    convert(dataset[0], dataset[1], out, size=4, workers=1, seed=0)

    # the written bands are the block mean of the composite turned into physical bands
    expected = block_mean(bands_from_composite(_image()), 2)
    for split in ("train", "test"):
        for stem in _stems(out / "spectral_bands" / split):
            bands = np.load(out / "spectral_bands" / split / f"{stem}.npy")
            assert bands.shape == (4, 4, 3)
            assert bands.dtype == np.float32
            np.testing.assert_allclose(bands, expected, atol=1e-6, err_msg="2x2 NaN-ignoring mean")


def test_convert_main_returns_zero(dataset: tuple[Path, Path], tmp_path: Path):
    from deepcontrail.mtg.convert import main

    out = tmp_path / "out"
    assert (
        main(
            [
                "--labels",
                str(dataset[0]),
                "--images",
                str(dataset[1]),
                "--out",
                str(out),
                "--size",
                "8",
                "--workers",
                "2",
            ]
        )
        == 0
    )


def test_convert_rejects_a_size_bigger_than_the_image(dataset: tuple[Path, Path], tmp_path: Path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="image side"):
        convert(dataset[0], dataset[1], out, size=16, workers=1)
