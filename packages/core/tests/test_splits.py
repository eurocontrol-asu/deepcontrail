"""Tests for deepcontrail.core.splits — train/val/fold partition and context frames."""

from __future__ import annotations

import pytest


def _stems(n: int) -> list[str]:
    return [f"{i:04d}" for i in range(n)]


def _partition(stems: list[str], split: str, **overrides: object) -> list[str]:
    from deepcontrail.core.splits import partition_stems

    kwargs: dict = {"seed": 42, "val_fraction": 0.1, "fold": None, "num_folds": 4} | overrides
    return partition_stems(stems, split, **kwargs)  # type: ignore[arg-type]


def test_partition_stems_test_split_returns_every_stem_sorted():
    stems = ["c", "a", "b"]

    result = _partition(stems, "test")

    assert result == ["a", "b", "c"], "test split must keep every stem, sorted"


def test_partition_stems_train_and_val_are_disjoint_and_cover_all():
    stems = _stems(50)

    train, val = _partition(stems, "train"), _partition(stems, "val")

    assert not set(train) & set(val), "a stem must never be in both train and val"
    assert sorted(train + val) == stems, "train and val together must cover every stem"


def test_partition_stems_val_size_follows_val_fraction():
    stems = _stems(50)

    val = _partition(stems, "val", val_fraction=0.2)

    assert len(val) == 10, "val must hold round(N * val_fraction) stems"


def test_partition_stems_val_keeps_at_least_one_stem():
    stems = _stems(3)

    val = _partition(stems, "val", val_fraction=0.01)

    assert len(val) == 1, "a tiny dataset must still yield one validation stem"


def test_partition_stems_same_seed_gives_same_partition():
    stems = _stems(50)

    first, second = _partition(stems, "val", seed=7), _partition(stems, "val", seed=7)

    assert first == second, "the partition must be deterministic for a given seed"


def test_partition_stems_different_seed_gives_different_partition():
    stems = _stems(50)

    first, second = _partition(stems, "val", seed=7), _partition(stems, "val", seed=8)

    assert first != second, "the seed must drive the random partition"


def test_partition_stems_ignores_input_order():
    stems = _stems(50)

    shuffled = list(reversed(stems))

    assert _partition(shuffled, "val") == _partition(stems, "val"), (
        "the partition must depend on the stems, not on directory listing order"
    )


@pytest.mark.parametrize("fold", [0, 1, 2])
def test_partition_stems_fold_assigns_every_kth_stem_to_val(fold: int):
    stems = _stems(9)

    val = _partition(stems, "val", fold=fold, num_folds=3)
    train = _partition(stems, "train", fold=fold, num_folds=3)

    assert val == stems[fold::3], "fold k validates on stems k, k+num_folds, ..."
    assert sorted(train + val) == stems, "train and val must cover every stem"
    assert not set(train) & set(val), "a stem must never be in both train and val"


def test_partition_stems_fold_takes_precedence_over_val_fraction():
    stems = _stems(9)

    val = _partition(stems, "val", fold=0, num_folds=3, val_fraction=0.5)

    assert val == stems[0::3], "val_fraction must be ignored when fold is set"


@pytest.mark.parametrize("temporal", [False, None])
def test_context_frames_are_none_unless_the_dataset_declares_them(temporal):
    from deepcontrail.core.splits import context_frames

    frames = context_frames(["tile_a1", "site_b2", "123", "123_b4"], temporal=temporal)

    assert frames == set(), "without declared context frames every id is a scene, as named"


def test_context_frames_of_a_temporal_dataset_are_suffixed_neighbours_of_a_scene():
    from deepcontrail.core.splits import context_frames

    frames = context_frames(["123", "123_b4", "123_b1", "123_a1", "x_a10", "x"], temporal=True)

    assert frames == {"123_b4", "123_b1", "123_a1", "x_a10"}


@pytest.mark.parametrize("stem", ["tile_a1", "a_b", "123_c1", "train_000001"])
def test_context_frames_need_their_scene(stem):
    from deepcontrail.core.splits import context_frames

    frames = context_frames([stem], temporal=True)

    assert frames == set(), f"{stem} has no scene it could be a context frame of, so it is a scene"


def test_split_fingerprint_ignores_order():
    from deepcontrail.core.splits import split_fingerprint

    assert split_fingerprint(["b", "a", "c"]) == split_fingerprint(["a", "b", "c"])


def test_split_fingerprint_differs_for_different_sets():
    from deepcontrail.core.splits import split_fingerprint

    assert split_fingerprint(["a", "b"]) != split_fingerprint(["a", "c"])


def test_split_fingerprint_is_short_hex():
    from deepcontrail.core.splits import split_fingerprint

    fingerprint = split_fingerprint(["a"])

    assert len(fingerprint) == 12
    int(fingerprint, 16)
