"""Tests for deepcontrail.mtg.split — whole-day grouping and the seeded split search."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pytest

from deepcontrail.mtg import labels


def _square() -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return a small square polygon (normalized) covering a few pixel centres at any size."""
    return ((0.25, 0.25), (0.4, 0.25), (0.4, 0.4), (0.25, 0.4))


def _sequence(
    name: str,
    project: str,
    zone: int,
    start: datetime,
    end: datetime,
    *,
    band: int = 4,
    n_frames: int = 2,
    low: bool = False,
) -> labels.Sequence:
    """Build a synthetic sequence with ``n_frames`` images, one High/Low contrail each."""
    frames = tuple(
        labels.LabelledFrame(
            title=f"{name}_f{i}",
            npy=f"frames/{name}_f{i}.npy",
            labelled_size=1024,
            contrails=tuple(
                labels.Contrail(
                    key=f"c{j}", polygons=_square(), confidence="Low" if low else "High"
                )
                for j in range(2)
            ),
        )
        for i in range(n_frames)
    )
    return labels.Sequence(
        name=name,
        project=project,
        utm_zone=zone,
        frame_band=band,
        start=start,
        end=end,
        frames=frames,
    )


def _three_projects() -> list[labels.Sequence]:
    """Three projects, each five day-groups; ``P0`` day 1 holds a merged pair of sequences."""
    seqs: list[labels.Sequence] = []
    # P0: zone 30, days 1-5, with day 1 being two overlapping sequences (a merged group).
    seqs += [
        _sequence("P0_a", "P0", 30, datetime(2025, 6, 1, 8, 0), datetime(2025, 6, 1, 16, 0)),
        _sequence("P0_b", "P0", 30, datetime(2025, 6, 1, 12, 0), datetime(2025, 6, 1, 20, 0)),
        _sequence("P0_c", "P0", 30, datetime(2025, 6, 2, 8, 0), datetime(2025, 6, 2, 16, 0)),
        _sequence("P0_d", "P0", 30, datetime(2025, 6, 3, 8, 0), datetime(2025, 6, 3, 16, 0)),
        _sequence("P0_e", "P0", 30, datetime(2025, 6, 4, 8, 0), datetime(2025, 6, 4, 16, 0)),
        _sequence("P0_f", "P0", 30, datetime(2025, 6, 5, 8, 0), datetime(2025, 6, 5, 16, 0)),
    ]
    # P1: zone 50, days 6-10. P2: zone 12, days 11-15. Distinct dates so no group ids collide.
    for day in range(6, 11):
        seqs.append(
            _sequence(
                f"P1_{day}", "P1", 50, datetime(2025, 6, day, 8, 0), datetime(2025, 6, day, 16, 0)
            )
        )
    for day in range(11, 16):
        seqs.append(
            _sequence(
                f"P2_{day}", "P2", 12, datetime(2025, 6, day, 8, 0), datetime(2025, 6, day, 16, 0)
            )
        )
    return seqs


def test_day_groups_puts_sequences_of_one_day_together():
    from deepcontrail.mtg.split import day_groups

    seqs = [
        _sequence("s1", "P", 30, datetime(2025, 6, 1, 8, 0), datetime(2025, 6, 1, 12, 0)),
        _sequence("s2", "P", 30, datetime(2025, 6, 1, 10, 0), datetime(2025, 6, 1, 14, 0)),
    ]

    groups = day_groups(seqs)

    assert groups == {"s1": "2025-06-01", "s2": "2025-06-01"}, "same-day sequences share one group"
    assert len(set(groups.values())) == 1


def test_day_groups_merges_overlapping_neighbours_across_midnight():
    from deepcontrail.mtg.split import day_groups

    seqs = [
        _sequence("late", "P", 27, datetime(2025, 6, 1, 23, 30), datetime(2025, 6, 2, 7, 30)),
        _sequence("early", "P", 28, datetime(2025, 6, 2, 7, 0), datetime(2025, 6, 2, 15, 0)),
    ]

    groups = day_groups(seqs)

    assert len(set(groups.values())) == 1, "zones 27/28 within 1 and overlapping merge"
    assert groups == {"late": "2025-06-01", "early": "2025-06-01"}, "group id is the earliest day"


def test_day_groups_keeps_distant_zones_apart():
    from deepcontrail.mtg.split import day_groups

    seqs = [
        _sequence("west", "P", 23, datetime(2025, 6, 1, 8, 0), datetime(2025, 6, 1, 16, 0)),
        _sequence("east", "P", 30, datetime(2025, 6, 2, 8, 0), datetime(2025, 6, 2, 16, 0)),
    ]

    groups = day_groups(seqs)

    assert len(set(groups.values())) == 2, "different days, non-overlapping -> two groups"
    assert set(groups.values()) == {"2025-06-01", "2025-06-02"}


def test_day_groups_raises_when_a_merged_group_mixes_projects():
    from deepcontrail.mtg.split import day_groups

    seqs = [
        _sequence("a", "P0", 30, datetime(2025, 6, 1, 8, 0), datetime(2025, 6, 1, 16, 0)),
        _sequence("b", "P1", 30, datetime(2025, 6, 1, 10, 0), datetime(2025, 6, 1, 18, 0)),
    ]

    with pytest.raises(ValueError, match="mixes projects"):
        day_groups(seqs)


def test_make_split_never_splits_a_group():
    from deepcontrail.mtg.split import day_groups, make_split

    seqs = _three_projects()
    groups = day_groups(seqs)
    assignment = make_split(seqs, groups, seed=0)

    merged = {groups["P0_a"]}
    assert merged == {groups["P0_b"]}, "the test builds P0 day 1 as one merged group"
    assert assignment["2025-06-01"] == assignment[groups["P0_a"]] == assignment[groups["P0_b"]], (
        "two sequences merged into one group must get the same split"
    )

    by_group: dict[str, list[str]] = defaultdict(list)
    for seq in seqs:
        by_group[groups[seq.name]].append(seq.name)
    for group_id, names in by_group.items():
        split = assignment[group_id]
        assert split in {"train", "val", "test"}
        assert {assignment[groups[name]] for name in names} == {split}, (
            "every sequence of a group resolves to that group's single split"
        )


def test_make_split_puts_at_least_one_group_per_project_in_test():
    from deepcontrail.mtg.split import day_groups, make_split

    seqs = _three_projects()
    groups = day_groups(seqs)
    assignment = make_split(seqs, groups, seed=0)

    for project in ("P0", "P1", "P2"):
        project_groups = {group_id for name, group_id in groups.items() if name.startswith(project)}
        assert any(assignment[group_id] == "test" for group_id in project_groups), (
            f"{project} keeps at least one whole day in the test holdout"
        )
    assert set(assignment) == set(groups.values()), "every group is assigned exactly once"


def test_make_split_is_deterministic_for_a_seed():
    from deepcontrail.mtg.split import day_groups, make_split

    seqs = _three_projects()
    groups = day_groups(seqs)

    first = make_split(seqs, groups, seed=7)
    second = make_split(seqs, groups, seed=7)
    other = make_split(seqs, groups, seed=8)

    assert first == second, "the same seed yields the same assignment"
    assert first != other, "a different seed can yield a different assignment"


def test_group_stats_count_only_frames_with_an_image():
    from deepcontrail.mtg.split import _group_stats, day_groups

    seq = labels.Sequence(
        name="s",
        project="P",
        utm_zone=30,
        frame_band=4,
        start=datetime(2025, 6, 1, 8, 0),
        end=datetime(2025, 6, 1, 12, 0),
        frames=(
            labels.LabelledFrame(
                "s_f0", "f0.npy", 1024, (labels.Contrail("c0", _square(), "High"),)
            ),
            labels.LabelledFrame("s_f1", None, 1024, (labels.Contrail("c1", _square(), "Low"),)),
        ),
    )

    stats = _group_stats((seq,), day_groups((seq,)))
    group_id = next(iter(stats))

    assert stats[group_id].frames == 1, "the frame without an image does not count"
    assert stats[group_id].contrails == 1
    assert stats[group_id].low == 0
