"""Day-aware train/val/test split for MTG/FCI sequences.

Frames 10 minutes apart are near-duplicates and one labelled sequence spans several hours,
so the split must hold out whole calendar days and keep sequences that overlap in space and
time on the same side. A *group* is a calendar day, merged (union-find) with every sequence
that overlaps another in space and time; :func:`make_split` then searches the assignments of
groups to splits and keeps the one that best matches the target fractions and contrast
statistics. The search is seeded, so it is deterministic for a given seed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepcontrail.mtg import labels


@dataclass(frozen=True)
class GroupStats:
    """Frame-weighted statistics of one day-group, used to score a split.

    Parameters
    ----------
    project
        The project the group's sequences belong to.
    frames
        Frames that have an image (``npy`` is not ``None``).
    contrails
        Contrails drawn on those frames. In several-labeller mode the per-frame count is
        AVERAGED over labellers, so it may be fractional; single-labeller counts are integers.
    low
        Contrails whose confidence is ``"Low"``; averaged over labellers in several-labeller
        mode, so it may be fractional too.
    """

    project: str
    frames: int
    contrails: float
    low: float


def day_groups(sequences: Sequence[labels.Sequence]) -> dict[str, str]:
    """Map each sequence name to a day-group id.

    A group is a calendar day — the ISO ``"YYYY-MM-DD"`` of a sequence's ``start`` — merged
    (union-find) with every sequence that overlaps another in space and time: the same
    ``frame_band``, an absolute ``utm_zone`` difference of at most one, and overlapping time
    windows (``start < other.end and other.start < end``). Sequences that overlap this way are
    near-duplicates that must not fall in different splits, even across midnight. The group id
    is the earliest day of the merged set.

    Parameters
    ----------
    sequences
        The labelled sequences to group.

    Returns
    -------
    dict[str, str]
        Sequence name -> group id (a calendar day). Two non-overlapping sequences that fall on
        the same calendar day share the day's id by design.

    Raises
    ------
    ValueError
        If a merged group mixes projects.
    """
    seqs = list(sequences)
    parent = list(range(len(seqs)))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(seqs)):
        a = seqs[i]
        for j in range(i + 1, len(seqs)):
            b = seqs[j]
            if (
                a.frame_band == b.frame_band
                and abs(a.utm_zone - b.utm_zone) <= 1
                and a.start < b.end
                and b.start < a.end
            ):
                union(i, j)

    components: dict[int, list[labels.Sequence]] = defaultdict(list)
    for i, seq in enumerate(seqs):
        components[find(i)].append(seq)

    result: dict[str, str] = {}
    for members in components.values():
        projects = {m.project for m in members}
        if len(projects) > 1:
            msg = (
                f"a day-group mixes projects {sorted(projects)}: {sorted(m.name for m in members)}"
            )
            raise ValueError(msg)
        group_id = min(m.start.date().isoformat() for m in members)
        for m in members:
            result[m.name] = group_id
    return result


def make_split(
    sequences: Sequence[labels.Sequence],
    groups: dict[str, str],
    *,
    test_fraction: float = 0.15,
    val_fraction: float = 0.10,
    seed: int = 0,
    draws: int = 20000,
) -> dict[str, str]:
    """Assign each group id to ``"train"``, ``"val"`` or ``"test"``.

    A seeded random search over ``draws`` assignments keeps the lowest-cost one (the first on
    ties), so the result is deterministic for a given ``seed``. Each draw, and for every
    project in sorted order, its groups are sorted then shuffled by the shared rng and split:
    one group to test when it has at most four, otherwise one or two; then zero to one to val
    when three remain, or one when more remain; the rest to train. This guarantees every
    project keeps at least one group in test.

    The cost of a draw is frame-weighted over all projects:

    ``|f_test - test_fraction| + |f_val - val_fraction|``
    ``+ 0.1 * |density_test / density_all - 1| + 0.1 * |density_val / density_all - 1|``
    ``+ 0.1 * |low_share_test - low_share_all|``

    where ``f`` is the frames fraction, ``density`` the contrails per frame and ``low_share``
    the share of ``"Low"`` contrails, so the split mirrors the data's volume and contrast.

    Parameters
    ----------
    sequences
        The labelled sequences, used for the per-group statistics.
    groups
        Sequence name -> group id, as returned by :func:`day_groups`.
    test_fraction, val_fraction
        Target fractions of frames in test and val.
    seed
        Seed for the random search.
    draws
        Number of assignments to try.

    Returns
    -------
    dict[str, str]
        Group id -> ``"train"`` | ``"val"`` | ``"test"``.
    """
    seqs = list(sequences)
    stats = _group_stats(seqs, groups)
    group_ids = list(stats)  # one entry per group id; ``groups`` maps a sequence to its id
    projects = sorted({stats[gid].project for gid in group_ids})
    project_groups = {
        p: sorted(gid for gid in group_ids if stats[gid].project == p) for p in projects
    }

    rng = np.random.default_rng(seed)
    best_assignment: dict[str, str] | None = None
    best_cost: float | None = None
    for _ in range(draws):
        assignment: dict[str, str] = {}
        for p in projects:
            order = sorted(project_groups[p])
            rng.shuffle(order)
            n = len(order)
            n_test = 1 if n <= 4 else int(rng.integers(1, 3))
            remaining = n - n_test
            if remaining < 3:
                n_val = 0
            elif remaining == 3:
                n_val = int(rng.integers(0, 2))
            else:
                n_val = 1
            for k, g in enumerate(order):
                if k < n_test:
                    assignment[g] = "test"
                elif k < n_test + n_val:
                    assignment[g] = "val"
                else:
                    assignment[g] = "train"
        cost = _cost(assignment, stats, test_fraction, val_fraction)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_assignment = dict(assignment)
    return best_assignment or {}


def split_report(
    sequences: Sequence[labels.Sequence],
    groups: dict[str, str],
    assignment: dict[str, str],
) -> dict:
    """Summarise an assignment, seed-independent, overall and per project.

    Parameters
    ----------
    sequences
        The labelled sequences, used for the per-group statistics.
    groups
        Sequence name -> group id.
    assignment
        Group id -> split, as returned by :func:`make_split`.

    Returns
    -------
    dict
        Per split (``train``/``val``/``test``): frames, frame fraction, contrails, contrails
        per frame, low share and group count; plus a ``per_project`` breakdown under the same
        keys. Every value is JSON-serialisable.
    """
    stats = _group_stats(sequences, groups)
    tot_frames = sum(s.frames for s in stats.values())
    tot_contrails = sum(s.contrails for s in stats.values())
    splits = ("train", "val", "test")
    report: dict = {
        sp: _split_numbers(
            [g for g, s in assignment.items() if s == sp],
            stats,
            tot_frames,
            tot_contrails,
        )
        for sp in splits
    }
    per_project: dict = {}
    for project in sorted({s.project for s in stats.values()}):
        proj_groups = [g for g in stats if stats[g].project == project]
        per_project[project] = {
            sp: _split_numbers(
                [g for g in proj_groups if assignment.get(g) == sp],
                stats,
                tot_frames,
                tot_contrails,
            )
            for sp in splits
        }
    report["per_project"] = per_project
    return report


def _group_stats(
    sequences: Sequence[labels.Sequence], groups: dict[str, str]
) -> dict[str, GroupStats]:
    """Frame-weighted statistics per group id (only frames with an image count)."""
    by_group: dict[str, list[labels.Sequence]] = defaultdict(list)
    for seq in sequences:
        by_group[groups[seq.name]].append(seq)
    stats: dict[str, GroupStats] = {}
    for gid, members in by_group.items():
        frames = 0
        contrails: float = 0
        low: float = 0
        for member in members:
            for frame in member.frames:
                if frame.npy is None:
                    continue
                frames += 1
                if frame.annotators is not None:
                    # several labellers drew on the frame: average their counts
                    counts = [len(annotator) for annotator in frame.annotators]
                    lows = [
                        sum(1 for c in annotator if c.confidence == "Low")
                        for annotator in frame.annotators
                    ]
                    contrails += sum(counts) / len(counts)
                    low += sum(lows) / len(lows)
                else:
                    contrails += len(frame.contrails)
                    low += sum(1 for c in frame.contrails if c.confidence == "Low")
        stats[gid] = GroupStats(
            project=members[0].project, frames=frames, contrails=contrails, low=low
        )
    return stats


def _cost(
    assignment: dict[str, str],
    stats: dict[str, GroupStats],
    test_fraction: float,
    val_fraction: float,
) -> float:
    """Frame-weighted distance of one assignment from the target fractions and contrast."""
    tot_frames = sum(s.frames for s in stats.values())
    tot_contrails = sum(s.contrails for s in stats.values())
    tot_low = sum(s.low for s in stats.values())
    if tot_frames == 0:
        return 1.0
    density_all = tot_contrails / tot_frames
    low_all = (tot_low / tot_contrails) if tot_contrails else 0.0

    def agg(split: str) -> tuple[int, float, float]:
        ids = [g for g, s in assignment.items() if s == split]
        frames = sum(stats[g].frames for g in ids)
        contrails = sum(stats[g].contrails for g in ids)
        low = sum(stats[g].low for g in ids)
        return frames, contrails, low

    tf, tc, tl = agg("test")
    vf, vc, _ = agg("val")
    f_test = tf / tot_frames
    f_val = vf / tot_frames

    cost = abs(f_test - test_fraction) + abs(f_val - val_fraction)
    if density_all:
        density_test = (tc / tf) if tf else density_all
        density_val = (vc / vf) if vf else density_all
        cost += 0.1 * abs(density_test / density_all - 1)
        cost += 0.1 * abs(density_val / density_all - 1)
    low_share_test = (tl / tc) if tc else low_all
    cost += 0.1 * abs(low_share_test - low_all)
    return cost


def _split_numbers(
    ids: list[str],
    stats: dict[str, GroupStats],
    tot_frames: int,
    tot_contrails: float,
) -> dict:
    """JSON-serialisable summary numbers for a set of group ids."""
    frames = sum(stats[i].frames for i in ids)
    contrails = sum(stats[i].contrails for i in ids)
    low = sum(stats[i].low for i in ids)
    return {
        "frames": frames,
        "fraction": round(frames / tot_frames, 6) if tot_frames else 0.0,
        "contrails": contrails,
        "contrails_per_frame": round(contrails / frames, 6) if frames else 0.0,
        "low_share": round(low / contrails, 6) if contrails else 0.0,
        "groups": len(ids),
    }
