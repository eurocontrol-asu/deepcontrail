"""Deterministic train/val/fold partition of scene stems, and context-frame detection."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def partition_stems(
    stems: list[str],
    split: Literal["train", "val", "test"],
    *,
    seed: int,
    val_fraction: float,
    fold: int | None,
    num_folds: int,
) -> list[str]:
    """Return the stems of ``split``, sorted.

    ``"test"`` returns every stem.  ``"train"`` and ``"val"`` partition the
    same stems deterministically:

    * When ``fold`` is set, k-fold CV partitioning (modular arithmetic)
      takes precedence.
    * Otherwise ``val_fraction`` determines the split — stems are
      shuffled with ``seed`` and the first ``val_fraction * N`` go to
      validation, the rest to training.
    """
    stems = sorted(stems)

    if split == "test":
        return stems

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
