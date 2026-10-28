"""Deterministic train/val/fold partition of scene stems, and context-frame detection."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

import numpy as np

_SEQUENCE_SUFFIX = re.compile(r"_[ab]\d+$")


def is_sequence_stem(stem: str) -> bool:
    """Return True if the stem has a temporal sequence suffix (e.g. ``_a1``, ``_b3``)."""
    return _SEQUENCE_SUFFIX.search(stem) is not None


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
