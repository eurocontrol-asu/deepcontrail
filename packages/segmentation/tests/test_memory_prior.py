"""Tests for deepcontrail.segmentation.memory_prior — corrupt_memory."""

from __future__ import annotations

import numpy as np

from deepcontrail.segmentation.config import MemoryCorruptionConfig
from deepcontrail.segmentation.memory_prior import corrupt_memory


def _none() -> MemoryCorruptionConfig:
    """Return a corruption config with every option off, so only the option under test fires."""
    return MemoryCorruptionConfig(
        empty_rate=0, drop_rate=0, max_shift=0, max_blur_sigma=0, min_scale=1
    )


def test_corrupt_memory_without_corruption_returns_the_memory():
    rng = np.random.default_rng(0)
    memory = rng.random((32, 32)).astype(np.float32)
    before = memory.copy()

    out = corrupt_memory(memory, _none(), np.random.default_rng(1))

    np.testing.assert_array_equal(out, memory, err_msg="no corruption keeps every value")
    assert out.dtype == np.float32, "the output is float32"
    assert memory[16, 7] == out[16, 7], "the values are unchanged with no corruption"
    # The input array must not be modified in place.
    np.testing.assert_array_equal(memory, before, err_msg="the input is not mutated")


def test_corrupt_memory_empty_rate_one_gives_zeros():
    memory = np.ones((16, 16), np.float32)
    config = _none().model_copy(update={"empty_rate": 1.0})

    out = corrupt_memory(memory, config, np.random.default_rng(0))

    assert np.count_nonzero(out) == 0, "empty_rate=1 zeroes the whole memory"
    assert out.shape == memory.shape


def test_corrupt_memory_drop_rate_one_removes_every_contrail():
    memory = np.zeros((32, 32), np.float32)
    memory[4:8, 4:8] = 1.0  # one blob
    memory[20:24, 20:24] = 1.0  # a second, separate blob
    config = _none().model_copy(update={"drop_rate": 1.0})

    out = corrupt_memory(memory, config, np.random.default_rng(0))

    assert np.count_nonzero(out) == 0, "drop_rate=1 removes every connected region"


def test_corrupt_memory_drop_rate_zero_keeps_every_contrail():
    memory = np.zeros((32, 32), np.float32)
    memory[4:8, 4:8] = 1.0
    memory[20:24, 20:24] = 1.0
    config = _none().model_copy(update={"drop_rate": 0.0})

    out = corrupt_memory(memory, config, np.random.default_rng(0))

    np.testing.assert_array_equal(out, memory, err_msg="drop_rate=0 removes nothing")


def test_corrupt_memory_shift_moves_a_pixel_within_max_shift():
    memory = np.zeros((32, 32), np.float32)
    memory[16, 16] = 1.0
    config = _none().model_copy(update={"max_shift": 3})

    for seed in range(20):
        out = corrupt_memory(memory, config, np.random.default_rng(seed))
        rows, cols = np.nonzero(out)

        assert rows.size == 1, (
            f"seed {seed}: the single contrail stays a single pixel after a shift"
        )
        assert abs(int(rows[0]) - 16) <= 3, (
            f"seed {seed}: the pixel moves by at most max_shift along the row axis"
        )
        assert abs(int(cols[0]) - 16) <= 3, (
            f"seed {seed}: the pixel moves by at most max_shift along the column axis"
        )


def test_corrupt_memory_shift_lets_zeros_enter_at_the_edges():
    memory = np.ones((16, 16), np.float32)
    config = _none().model_copy(update={"max_shift": 4})

    out = None
    for seed in range(50):
        candidate = corrupt_memory(memory, config, np.random.default_rng(seed))
        if np.any(candidate == 0.0) and np.any(candidate == 1.0):
            out = candidate  # a seed that actually shifted (a non-zero dy or dx)
            break

    assert out is not None, "some seed in 0..49 produces a non-zero shift"
    assert 0.0 in np.unique(out), "a shift brings zeros in at the edges"
    assert 1.0 in np.unique(out), "the pixels that did not move stay at 1"


def test_corrupt_memory_scale_stays_between_min_scale_and_one():
    memory = np.ones((16, 16), np.float32)
    config = _none().model_copy(update={"min_scale": 0.6})

    for seed in range(10):
        out = corrupt_memory(memory, config, np.random.default_rng(seed))

        assert out.min() >= 0.6, f"seed {seed}: scaling never drops a value below min_scale"
        assert out.max() <= 1.0, f"seed {seed}: scaling never raises a value above 1"


def test_corrupt_memory_output_stays_in_unit_range_with_every_corruption():
    rng = np.random.default_rng(3)
    memory = rng.random((32, 32)).astype(np.float32)
    config = MemoryCorruptionConfig()  # every option at its default, all firing

    for seed in range(20):
        out = corrupt_memory(memory, config, np.random.default_rng(seed))

        assert out.dtype == np.float32, "the output is float32"
        assert out.shape == memory.shape, "the output keeps the memory's shape"
        assert out.min() >= 0.0, "the output never goes below 0"
        assert out.max() <= 1.0, "the output never goes above 1"


def test_corrupt_memory_is_reproducible_with_the_same_seed():
    memory = np.ones((16, 16), np.float32)
    memory[3:6, 3:6] = 0.5
    config = MemoryCorruptionConfig()

    a = corrupt_memory(memory, config, np.random.default_rng(7))
    b = corrupt_memory(memory, config, np.random.default_rng(7))

    np.testing.assert_array_equal(a, b, err_msg="the same seed gives the same corruption")


def _line() -> np.ndarray:
    """Return a (64, 64) memory with one contrail, 1 px wide, at row 32, columns 20-43."""
    memory = np.zeros((64, 64), np.float32)
    memory[32, 20:44] = 1.0
    return memory


def test_corrupt_memory_ghost_adds_a_shifted_copy_and_keeps_the_original():
    memory = _line()
    config = _none().model_copy(update={"ghost_rate": 1.0, "ghost_shift": (12, 12)})

    out = corrupt_memory(memory, config, np.random.default_rng(0))

    assert (out[memory > 0] == 1.0).all(), "the true contrail stays in the memory"
    assert (out > 0).sum() > (memory > 0).sum(), "a false copy is added where the image has none"

    def shifted(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
        moved = np.zeros_like(image)
        h, w = image.shape
        moved[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = image[
            max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
        ]
        return moved

    matches = [
        (dy, dx)
        for dy in range(-13, 14)
        for dx in range(-13, 14)
        if 11 <= np.hypot(dy, dx) <= 13
        and np.array_equal(out, np.maximum(memory, shifted(memory, dy, dx)))
    ]
    assert matches, "the output is the contrail plus one copy of it shifted by about 12 px"


def test_corrupt_memory_ghost_rate_zero_leaves_the_random_stream_unchanged():
    memory = np.random.default_rng(3).uniform(0, 1, (32, 32)).astype(np.float32)
    plain = MemoryCorruptionConfig()
    explicit = MemoryCorruptionConfig(ghost_rate=0.0, max_dilation=0)

    a = corrupt_memory(memory, plain, np.random.default_rng(7))
    b = corrupt_memory(memory, explicit, np.random.default_rng(7))

    np.testing.assert_array_equal(
        a, b, err_msg="ghost and dilation off draw nothing (TM1 unchanged)"
    )


def test_corrupt_memory_dilation_widens_the_contrails():
    memory = _line()
    config = _none().model_copy(update={"max_dilation": 2})

    widened = [corrupt_memory(memory, config, np.random.default_rng(s)) for s in range(10)]

    assert all((w[memory > 0] == 1.0).all() for w in widened), "the contrail itself stays"
    assert any((w > 0).sum() > (memory > 0).sum() for w in widened), "some draws widen it"
    assert all(np.nonzero(w)[0].min() >= 30 for w in widened), "by at most 2 px"


def test_memory_corruption_rejects_a_ghost_shift_range_upside_down():
    import pytest

    with pytest.raises(ValueError, match="ghost_shift"):
        MemoryCorruptionConfig(ghost_shift=(40, 10))
