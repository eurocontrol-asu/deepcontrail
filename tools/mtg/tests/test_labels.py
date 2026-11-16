"""Tests for deepcontrail.mtg.labels — reading an Encord label export."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_sequence_name_reads_both_naming_styles():
    from deepcontrail.mtg.labels import parse_sequence_name

    mtg = parse_sequence_name("MTG_202506090530_202506091330_zone31_frame4")
    mti1 = parse_sequence_name("MTI1_20250205_1540_zone25_frame3")

    assert mtg[0] == datetime(2025, 6, 9, 5, 30), "MTG_<start>_<end>_zone.._frame.."
    assert mtg[1] == datetime(2025, 6, 9, 13, 30), "MTG names carry an explicit end"
    assert (mtg[2], mtg[3]) == (31, 4), "utm_zone and frame_band come from the name"
    assert mti1[0] == datetime(2025, 2, 5, 15, 40), "MTI1_<date>_<time>_zone.._frame.."
    assert mti1[1] is None, "MTI1 names carry no end time"
    assert (mti1[2], mti1[3]) == (25, 3)


def test_parse_sequence_name_rejects_other_names():
    from deepcontrail.mtg.labels import parse_sequence_name

    for bad in ("MTG_202506090530_zone31_frame4", "MTI1_20250205_zone25_frame3", "foo_123"):
        with pytest.raises(ValueError, match="sequence name"):
            parse_sequence_name(bad)


def test_contrail_confidence_takes_the_highest_level_over_tracks():
    from deepcontrail.mtg.labels import contrail_confidence

    label = {
        "contrail_groups": {"c1": ["t1", "t2", "t3"]},
        "tracks": {
            "t1": {"static": {"Confidence Level": "Low"}},
            "t2": {"static": {"Confidence Level": "High"}},
            "t3": {"static": {"Confidence Level": "Medium"}},
        },
    }

    assert contrail_confidence("c1", label) == "High", "the highest grouped track wins"


def test_contrail_confidence_is_none_without_a_known_level():
    from deepcontrail.mtg.labels import contrail_confidence

    label = {
        "contrail_groups": {"c1": ["t1", "t2"], "c2": ["t3"]},
        "tracks": {
            "t1": {"static": {"Other": "High"}},  # the "Confidence Level" key is missing
            "t2": {"track": 1},  # "static" is missing
            "t3": {"static": {"Confidence Level": "Extreme"}},  # an unknown value
        },
    }

    assert contrail_confidence("c1", label) is None, "no track carries a known level"
    assert contrail_confidence("c2", label) is None


def test_read_sequence_merges_fragments_into_one_contrail(tmp_path: Path):
    from deepcontrail.mtg.labels import read_sequence

    path = _write_sequence(
        tmp_path,
        name="MTI1_20250205_1540_zone25_frame3",
        images=[
            {
                "image_title": "MTI1_ash_22_12_1_202502051540_zone25_frame3.png",
                "image_npy": "frames/0000.npy",
                "width": 1024,
                "height": 1024,
                "contrails": {
                    "c1": [
                        [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
                        [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4]],
                    ]
                },
            }
        ],
        groups={"c1": ["t1"]},
        tracks={"t1": {"static": {"Confidence Level": "High"}}},
    )

    seq = read_sequence(path)

    assert len(seq.frames) == 1
    frame = seq.frames[0]
    assert len(frame.contrails) == 1, "two fragments under one key are one Contrail"
    contrail = frame.contrails[0]
    assert contrail.key == "c1"
    assert len(contrail.polygons) == 2
    assert contrail.confidence == "High", "read from contrail_groups/c1 -> tracks/t1"
    assert frame.title == "MTI1_ash_22_12_1_202502051540_zone25_frame3"
    assert frame.npy == "frames/0000.npy"
    assert frame.labelled_size == 1024


def test_read_sequence_drops_polygons_with_fewer_than_three_points(tmp_path: Path):
    from deepcontrail.mtg.labels import read_sequence

    path = _write_sequence(
        tmp_path,
        name="MTI1_20250205_1540_zone25_frame3",
        images=[
            {
                "image_title": "frame0.png",
                "width": 512,
                "height": 512,
                "contrails": {
                    "c1": [[[0.1, 0.1], [0.2, 0.2]]],  # two points -> the polygon is dropped
                    "c2": [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]],  # three points -> kept
                },
            }
        ],
    )

    seq = read_sequence(path)

    contrails = seq.frames[0].contrails
    assert [c.key for c in contrails] == ["c2"], "a contrail left without polygons is dropped"
    assert len(contrails[0].polygons) == 1


def test_read_sequence_defaults_end_to_eight_hours_after_start_for_mti1_names(tmp_path: Path):
    from deepcontrail.mtg.labels import read_sequence

    path = _write_sequence(tmp_path, name="MTI1_20250205_1540_zone25_frame3", images=[])

    seq = read_sequence(path)

    assert seq.start == datetime(2025, 2, 5, 15, 40)
    assert seq.end == datetime(2025, 2, 5, 23, 40), "MTI1 names default the end to start + 8 h"
    assert seq.frames == ()


def _write_sequence(
    tmp_path: Path,
    *,
    name: str,
    images: list[dict],
    project: str = "MTG_S10_L3",
    utm_zone: str = "25",
    frame_band: str = "MTI1",
    groups: dict | None = None,
    tracks: dict | None = None,
) -> Path:
    """Write one Encord-style label export to ``tmp_path`` and return its path."""
    data = {
        "sequence": name,
        "project_name": project,
        "utm_zone": utm_zone,
        "frame_band": frame_band,
        "images": images,
        "contrail_groups": groups or {},
        "tracks": tracks or {},
    }
    path = tmp_path / "label.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
