"""Deterministic train/val/fold partition of scene stems, and context-frame detection."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

# ``<scene>_b<k>`` / ``<scene>_a<k>``: frame k steps before / after a scene, in temporal datasets.
_CONTEXT_FRAME = re.compile(r"(?P<scene>.+)_[ab]\d+")


def context_frames(stems: Iterable[str], *, temporal: bool | None) -> set[str]:
    """Return the stems that are context frames rather than scenes.

    Context frames exist only in datasets that declare them (``temporal: true`` in
    ``metadata/dataset_info.json``). There, a stem is a context frame when it is
    ``<scene>_b<k>`` or ``<scene>_a<k>`` for a scene present among ``stems``; any other stem
    is a scene. Otherwise every stem is a scene and its id is used as written, so ids such
    as ``tile_a1`` are ordinary scenes.

    Parameters
    ----------
    stems
        File stems of one split (``spectral_bands/<split>/*.npy``).
    temporal
        The dataset's ``temporal`` field; ``None`` (not declared) means no context frames.

    Returns
    -------
    set[str]
        The stems that are context frames.
    """
    if not temporal:
        return set()
    present = set(stems)
    return {
        stem
        for stem in present
        if (match := _CONTEXT_FRAME.fullmatch(stem)) and match["scene"] in present
    }


def read_stem_list(path: str | Path) -> list[str]:
    """Read a list of scene ids: one per line; blank and ``#`` lines are skipped."""
    ids: list[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        id_ = line.strip()
        if not id_ or id_.startswith("#"):
            continue
        if id_ in seen:
            msg = f"duplicate id {id_!r} in {path}"
            raise ValueError(msg)
        seen.add(id_)
        ids.append(id_)
    return ids


def partition_stems(
    stems: list[str],
    split: Literal["train", "val", "test"],
    *,
    seed: int,
    val_fraction: float,
    fold: int | None,
    num_folds: int,
    val_stems: Collection[str] | None = None,
    train_stems: Collection[str] | None = None,
) -> list[str]:
    """Return the stems of ``split``, sorted.

    ``"test"`` returns every stem.  ``"train"`` and ``"val"`` partition the
    same stems deterministically:

    * When ``fold`` is set, k-fold CV partitioning (modular arithmetic)
      takes precedence.
    * Otherwise ``val_fraction`` determines the split — stems are
      shuffled with ``seed`` and the first ``val_fraction * N`` go to
      validation, the rest to training.

    When ``val_stems`` is set (for ``"train"``/``"val"`` only), it replaces the
    above: the validation split is exactly those scene ids, in sorted order, and
    training is their complement. Every id of ``val_stems`` must be among
    ``stems``. ``fold``, ``val_fraction`` and ``seed`` are then ignored — useful
    for a fixed validation set of whole days, which a random frame-level split
    would leak near-duplicates from.

    When ``train_stems`` is set (for ``"train"`` only), it restricts training to
    exactly those scene ids, in sorted order — useful for a learning curve on a
    fraction of the data. Every id of ``train_stems`` must be among ``stems`` and
    must not be a validation stem (``val_stems``), or a ``ValueError`` is raised;
    ``val``, ``test`` and the validation split are unaffected.
    """
    stems = sorted(stems)

    if split == "test":
        return stems

    if train_stems is not None and split == "train":
        present = set(stems)
        val = set(val_stems) if val_stems is not None else set()
        missing = [s for s in train_stems if s not in present]
        if missing:
            msg = f"train_stems reference ids not in the dataset: {missing[:5]}"
            raise ValueError(msg)
        clashes = [s for s in train_stems if s in val]
        if clashes:
            msg = f"train_stems overlap the validation split: {clashes[:5]}"
            raise ValueError(msg)
        keep = set(train_stems)
        return [s for s in stems if s in keep]

    if val_stems is not None:
        val = set(val_stems)
        present = set(stems)
        missing = [s for s in val_stems if s not in present]
        if missing:
            msg = f"val_stems reference ids not in the dataset: {missing[:5]}"
            raise ValueError(msg)
        if split == "val":
            return [s for s in stems if s in val]
        return [s for s in stems if s not in val]

    if fold is not None:
        if split == "val":
            return [s for i, s in enumerate(stems) if i % num_folds == fold]
        return [s for i, s in enumerate(stems) if i % num_folds != fold]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(stems))
    n_val = max(1, round(len(stems) * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    if split == "val":
        return [s for i, s in enumerate(stems) if i in val_idx]
    return [s for i, s in enumerate(stems) if i not in val_idx]


def split_fingerprint(stems: list[str]) -> str:
    """Return a 12-hex-digit fingerprint of a set of stems (independent of their order).

    Two runs used the same records for a split exactly when the fingerprints match.
    """
    return hashlib.sha256("\n".join(sorted(stems)).encode()).hexdigest()[:12]
